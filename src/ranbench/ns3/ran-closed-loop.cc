// C2 closed-loop radio scenario for EXP-2026-003 (ns-3.48 + 5G-LENA v5.1).
//
// 21-cell TR 38.901 UMa layout (7 sites x 3 sectors, ISD 500 m) or a 1-cell instrument layout,
// four DL traffic classes, an enforcement schedule (E2 control installs) and a numerical xApp.
// Traffic "type" per class: cbr and periodic send one packet_size packet every
// packet_size*8/rate_kbps seconds (the same fixed-interval generator; periodic names the low-rate
// IoT use); backlogged ignores rate_kbps and sends packet_size packets open-loop at 8 bit per
// resource element of the whole carrier (7.47 bit/s/Hz x bandwidth, 74.7 Mb/s at 10 MHz), above
// any single-layer DL rate, so the UE's RLC buffer stays full (full-buffer best effort). Every
// type is open-loop, so tx_trace.csv does not depend on the schedule (C-4).
// Actuation: NrMacSchedulerLC::m_priority of each UE's class flow (5QI 80 dedicated QoS flow) in
// a NrMacSchedulerTdmaQos subclass; the QoS weight is proportional to (100 - m_priority).
//
// Output files (all in --outputDir):
//   ue_slots.csv      per 100 ms window (t_slot-0.1, t_slot] and UE
//   tx_trace.csv      per 100 ms window and UE: packets/bytes sent by the source (C-4 trace)
//   positions.csv     every 100 ms
//   cell_prb.csv      per 100 ms window and cell, from NrGnbPhy "SlotDataStats"
//   handover.csv      one row per handover (NrGnbRrc HandoverStart, NrUeRrc HandoverEndOk,
//                     NrGnbRrc HandoverTotalTime, NrUeRrc HandoverEndError)
//   rlf.csv           NrUeRrc RadioLinkFailureCause
//   enforcement.csv   every change of a (cell, class) m_priority level, source policy|xapp|interpreter
//   lc_priority.csv   every 100 ms: m_priority read back from each scheduler's LC objects
//   cells.csv, ue_map.csv, streams.json   layout, UE->class map, RNG stream map

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/hexagonal-grid-scenario-helper.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-json.hpp"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/seq-ts-header.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace ns3;
using json = nlohmann::json;

NS_LOG_COMPONENT_DEFINE("RanClosedLoop");

// 5QI of the per-UE class flow (dedicated QoS flow). The default flow (5QI 9) and the SRBs
// (5QI 5 / 1) carry no scenario traffic and are never touched.
static const uint8_t CLASS_FLOW_5QI = NrQosFlow::NGBR_LOW_LAT_EMBB; // 80

// RNG stream bases, one disjoint block per component (asserted non-overlapping at run time).
static const int64_t STREAM_MOBILITY = 1000;    // UE drop (2) + per-UE mobility models (3 each)
static const int64_t STREAM_CHANNEL = 50000;    // propagation loss, channel condition, fading
static const int64_t STREAM_TRAFFIC = 60000;    // traffic start jitter
static const int64_t STREAM_GNB_DEV = 100000;   // everything else: gNB PHY/scheduler
static const int64_t STREAM_UE_DEV = 200000;    // everything else: UE PHY/MAC
static const int64_t STREAM_EPC = 300000;       // everything else: EPC
static const int64_t STREAM_REMOTE = 301000;    // everything else: remote host stack
static const int64_t STREAM_UE_STACK = 302000;  // everything else: UE internet stacks
static const int64_t STREAM_BLOCK_END = 400000;

/**
 * TDMA QoS scheduler whose per-flow m_priority can be changed at run time.
 * The scheduler resolves RNTI -> IMSI through its own gNB RRC and asks the scenario which
 * m_priority the UE's class flow must have in this cell, both when a flow is configured
 * (attach, handover in, re-establishment) and when a level changes.
 */
class C2QosScheduler : public NrMacSchedulerTdmaQos
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::C2QosScheduler")
                                .SetParent<NrMacSchedulerTdmaQos>()
                                .AddConstructor<C2QosScheduler>();
        return tid;
    }

    uint16_t m_cellId{0};
    NrGnbRrc* m_rrc{nullptr};                                     // non-owning
    std::function<int(uint16_t cellId, uint64_t imsi)> m_lookup; // -1 = not a scenario UE
    std::map<uint16_t, std::shared_ptr<NrMacSchedulerUeInfo>> m_registeredUes;

    uint64_t ImsiOf(uint16_t rnti) const
    {
        if (m_rrc && m_rrc->HasUeManager(rnti))
        {
            return m_rrc->GetUeManager(rnti)->GetImsi();
        }
        return 0;
    }

    // Set m_priority of the class flow LC(s) of one UE; returns the number of LCs changed.
    uint32_t SetFlowPriority(uint16_t rnti, uint8_t prio)
    {
        uint32_t n = 0;
        auto it = m_registeredUes.find(rnti);
        if (it == m_registeredUes.end())
        {
            return 0;
        }
        for (auto& [lcgId, lcg] : it->second->m_dlLCG)
        {
            for (uint8_t lcId : lcg->GetLCId())
            {
                auto& lc = lcg->GetLC(lcId);
                if (lc->m_fiveQi == CLASS_FLOW_5QI)
                {
                    lc->m_priority = prio;
                    ++n;
                }
            }
        }
        return n;
    }

    // Read back m_priority of the class flow LC; -1 if not configured.
    int GetFlowPriority(uint16_t rnti) const
    {
        auto it = m_registeredUes.find(rnti);
        if (it == m_registeredUes.end())
        {
            return -1;
        }
        for (auto& [lcgId, lcg] : it->second->m_dlLCG)
        {
            for (uint8_t lcId : lcg->GetLCId())
            {
                auto& lc = lcg->GetLC(lcId);
                if (lc->m_fiveQi == CLASS_FLOW_5QI)
                {
                    return lc->m_priority;
                }
            }
        }
        return -1;
    }

  protected:
    std::shared_ptr<NrMacSchedulerUeInfo> CreateUeRepresentation(
        const NrMacCschedSapProvider::CschedUeConfigReqParameters& params) const override
    {
        auto ueInfo = NrMacSchedulerTdmaQos::CreateUeRepresentation(params);
        const_cast<C2QosScheduler*>(this)->m_registeredUes[params.m_rnti] = ueInfo;
        return ueInfo;
    }

    void DoCschedUeReleaseReq(
        const NrMacCschedSapProvider::CschedUeReleaseReqParameters& params) override
    {
        m_registeredUes.erase(params.m_rnti);
        NrMacSchedulerTdmaQos::DoCschedUeReleaseReq(params);
    }

    void DoCschedLcConfigReq(
        const NrMacCschedSapProvider::CschedLcConfigReqParameters& params) override
    {
        NrMacSchedulerTdmaQos::DoCschedLcConfigReq(params);
        if (!m_lookup)
        {
            return;
        }
        int prio = m_lookup(m_cellId, ImsiOf(params.m_rnti));
        if (prio >= 0)
        {
            SetFlowPriority(params.m_rnti, static_cast<uint8_t>(prio));
        }
    }
};

NS_OBJECT_ENSURE_REGISTERED(C2QosScheduler);

/**
 * UE-side sink. Reads the SeqTsHeader send timestamp of every packet (one-way delay) and keeps
 * 100 ms slot counters and last-1 s xApp window counters.
 */
class C2TrafficSink : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ns3::C2TrafficSink").SetParent<Application>().AddConstructor<C2TrafficSink>();
        return tid;
    }

    void Setup(uint32_t ueIdx, uint16_t port)
    {
        m_ueIdx = ueIdx;
        m_port = port;
    }

    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_port));
        m_socket->SetRecvCallback(MakeCallback(&C2TrafficSink::HandleRead, this));
    }

    void StopApplication() override
    {
        if (m_socket)
        {
            m_socket->Close();
            m_socket = nullptr;
        }
    }

    void HandleRead(Ptr<Socket> socket);

    uint32_t m_ueIdx{0};
    uint16_t m_port{0};
    Ptr<Socket> m_socket;
    uint64_t m_slotRxBytes{0};
    uint32_t m_slotRxPkts{0};
    std::vector<double> m_slotDelaysMs;
};

NS_OBJECT_ENSURE_REGISTERED(C2TrafficSink);

struct UeFlowSpec
{
    uint64_t imsi{0};
    Ipv4Address destIp;
    uint16_t destPort{0};
    uint32_t pktSize{1400};
    Time interval{MilliSeconds(10)};
    Time startTime{Seconds(0.2)};
    uint32_t seq{0};
    EventId sendEvent;
    uint64_t slotTxBytes{0};
    uint32_t slotTxPkts{0};
};

/**
 * Remote-host source: one open-loop UDP flow per UE (cbr, periodic or backlogged), each packet stamped by SeqTsHeader
 * (the header records Simulator::Now() when it is constructed, i.e. at send).
 */
class C2TrafficSource : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::C2TrafficSource")
                                .SetParent<Application>()
                                .AddConstructor<C2TrafficSource>();
        return tid;
    }

    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        for (size_t i = 0; i < m_flows.size(); ++i)
        {
            Time delay = m_flows[i].startTime - Simulator::Now();
            m_flows[i].sendEvent =
                Simulator::Schedule(delay, &C2TrafficSource::SendPacket, this, i);
        }
    }

    void StopApplication() override
    {
        for (auto& flow : m_flows)
        {
            flow.sendEvent.Cancel();
        }
        if (m_socket)
        {
            m_socket->Close();
            m_socket = nullptr;
        }
    }

    void SendPacket(size_t idx);

    Ptr<Socket> m_socket;
    std::vector<UeFlowSpec> m_flows;
};

NS_OBJECT_ENSURE_REGISTERED(C2TrafficSource);

// ---------------------------------------------------------------------------------------------
// Global scenario state
// ---------------------------------------------------------------------------------------------
static const std::vector<std::string> ALL_CLASSES = {"video", "xr", "iot", "be"};

static std::ofstream g_ueSlotsOut, g_txTraceOut, g_positionsOut, g_cellPrbOut, g_handoverOut,
    g_rlfOut, g_enforcementOut, g_lcPrioOut;

static std::map<uint16_t, uint64_t> g_cellUsedReg;  // RB x symbol (DL+UL data)
static std::map<uint16_t, uint64_t> g_cellAvailReg; // RB x non-CTRL symbols

static std::map<uint16_t, Ptr<NrGnbNetDevice>> g_cellGnbDevs;
static std::map<uint16_t, Ptr<C2QosScheduler>> g_cellSched;
static std::vector<Ptr<NrUeNetDevice>> g_ueDevs;
static std::vector<Ptr<Node>> g_ueNodes;
static std::vector<Ptr<C2TrafficSink>> g_sinks;
static std::vector<std::string> g_ueClass;       // by UE index
static std::map<uint64_t, uint32_t> g_imsiToIdx; // IMSI -> UE index
static Ptr<C2TrafficSource> g_source;             // flows indexed by UE index

static std::map<std::string, std::vector<uint16_t>> g_clusters;
static std::map<std::string, uint8_t> g_priorityLevels;
static uint8_t g_defaultPriority = 50;
static std::map<uint16_t, std::map<std::string, uint8_t>> g_basePriority;    // active policy
static std::map<uint16_t, std::map<std::string, uint8_t>> g_currentPriority; // policy + xApp

struct ClassTarget
{
    std::string type;
    double rateKbps{0.0};
    uint32_t packetSize{1400};
    std::string targetType; // throughput_floor_kbps | delay_p95_ms
    double targetValue{0.0};
};

static std::map<std::string, ClassTarget> g_classTargets;
static uint8_t g_xappStep = 2;
static double g_xappMargin = 0.10;
static uint8_t g_xappBand = 10;
static bool g_xappEnabled = true;
static bool g_rq5Enabled = false;
static int g_rq5Socket = -1;
static double g_rq5PendingUntil = -1.0;
static const double RQ5_DELTA_E2_S = 0.005;

// xApp counters of the current 1 s window, filled at each send/receive event and attributed to
// the cell serving the UE at that event (UE RRC in a CONNECTED_* state; none otherwise).
struct XappCellClassBucket
{
    std::map<uint32_t, uint32_t> txPktsByUe; // UE index -> packets sent while served by this cell
    uint64_t rxBytes{0};
    std::vector<double> delaysMs;

    void Reset()
    {
        txPktsByUe.clear();
        rxBytes = 0;
        delaysMs.clear();
    }
};

static std::map<uint16_t, std::map<std::string, XappCellClassBucket>> g_xappBuckets;
static std::vector<uint32_t> g_xappUeTxPkts; // UE index -> packets sent in the window (any state)
static std::map<uint16_t, std::map<std::string, uint64_t>> g_rq5DlSymbols;
static std::map<uint16_t, uint64_t> g_rq5CellAvailSymbols;

// Serving cell of UE idx now, 0 when the UE is not connected (idle, attaching, after RLF).
static uint16_t
ServingCell(uint32_t idx)
{
    Ptr<NrUeRrc> rrc = g_ueDevs[idx]->GetRrc();
    NrUeRrc::State state = rrc->GetState();
    return state == NrUeRrc::CONNECTED_NORMALLY || state == NrUeRrc::CONNECTED_HANDOVER
               ? rrc->GetCellId()
               : 0;
}

void
C2TrafficSink::HandleRead(Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from)))
    {
        uint32_t pktBytes = packet->GetSize();
        SeqTsHeader header;
        if (pktBytes < header.GetSerializedSize())
        {
            continue;
        }
        packet->RemoveHeader(header);
        double delayMs = (Simulator::Now() - header.GetTs()).GetSeconds() * 1000.0;
        m_slotRxBytes += pktBytes;
        m_slotRxPkts++;
        m_slotDelaysMs.push_back(delayMs);

        uint16_t cellId = ServingCell(m_ueIdx);
        if (cellId != 0)
        {
            auto& b = g_xappBuckets[cellId][g_ueClass[m_ueIdx]];
            b.rxBytes += pktBytes;
            b.delaysMs.push_back(delayMs);
        }
    }
}

void
C2TrafficSource::SendPacket(size_t idx)
{
    auto& flow = m_flows[idx];
    SeqTsHeader header;
    header.SetSeq(flow.seq++);
    uint32_t hs = header.GetSerializedSize();
    Ptr<Packet> p = Create<Packet>(flow.pktSize > hs ? flow.pktSize - hs : 0);
    p->AddHeader(header);
    m_socket->SendTo(p, 0, InetSocketAddress(flow.destIp, flow.destPort));
    flow.slotTxBytes += p->GetSize();
    flow.slotTxPkts++;

    g_xappUeTxPkts[idx]++;
    uint16_t cellId = ServingCell(idx);
    if (cellId != 0)
    {
        g_xappBuckets[cellId][g_ueClass[idx]].txPktsByUe[idx]++;
    }

    flow.sendEvent = Simulator::Schedule(flow.interval, &C2TrafficSource::SendPacket, this, idx);
}

// Priority the scheduler must apply to a UE's class flow in a cell (-1: not a scenario UE).
static int
LookupPriority(uint16_t cellId, uint64_t imsi)
{
    auto it = g_imsiToIdx.find(imsi);
    if (it == g_imsiToIdx.end())
    {
        return -1;
    }
    return g_currentPriority[cellId][g_ueClass[it->second]];
}

static std::string
Fmt(double v, int prec)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

// Nearest-rank p95 (sorts in place).
static double
P95(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::ceil(0.95 * v.size())) - 1;
    return v[idx];
}

// ---------------------------------------------------------------------------------------------
// Trace callbacks
// ---------------------------------------------------------------------------------------------
static void
OnPhySlotDataStats(const SfnSf&,
                   uint32_t,
                   uint32_t usedReg,
                   uint32_t,
                   uint32_t availableRb,
                   uint32_t availableSym,
                   uint16_t,
                   uint16_t cellId)
{
    // usedReg is in units of 1 RB x 1 symbol (nr-gnb-phy.h SlotStatsTracedCallback)
    g_cellUsedReg[cellId] += usedReg;
    g_cellAvailReg[cellId] += static_cast<uint64_t>(availableRb) * availableSym;
    if (g_rq5Enabled)
    {
        g_rq5CellAvailSymbols[cellId] += availableSym;
    }
}

static void
OnDlScheduling(uint16_t cellId, NrSchedulingCallbackInfo info)
{
    auto itSched = g_cellSched.find(cellId);
    if (itSched == g_cellSched.end())
    {
        return;
    }
    uint64_t imsi = itSched->second->ImsiOf(info.m_rnti);
    auto itIdx = g_imsiToIdx.find(imsi);
    if (itIdx != g_imsiToIdx.end())
    {
        g_rq5DlSymbols[cellId][g_ueClass[itIdx->second]] += info.m_numSym;
    }
}

struct PendingHo
{
    double tStart{-1};
    double tEndOk{-1};
    uint16_t source{0};
    uint16_t target{0};
};

static std::map<uint64_t, PendingHo> g_pendingHo;

static void
EmitHo(uint64_t imsi, const PendingHo& h, const std::string& totalMs, const std::string& outcome)
{
    g_handoverOut << (h.tStart >= 0 ? Fmt(h.tStart, 6) : "") << ","
                  << (h.tEndOk >= 0 ? Fmt(h.tEndOk, 6) : "") << "," << imsi << "," << h.source
                  << "," << h.target << "," << totalMs << "," << outcome << "\n";
}

static void
OnGnbHandoverStart(uint64_t imsi, uint16_t cellId, uint16_t, uint16_t targetCellId)
{
    auto it = g_pendingHo.find(imsi);
    if (it != g_pendingHo.end())
    {
        EmitHo(imsi, it->second, "", "superseded");
    }
    PendingHo h;
    h.tStart = Simulator::Now().GetSeconds();
    h.source = cellId;
    h.target = targetCellId;
    g_pendingHo[imsi] = h;
}

static void
OnUeHandoverEndOk(uint64_t imsi, uint16_t cellId, uint16_t)
{
    auto it = g_pendingHo.find(imsi);
    if (it != g_pendingHo.end())
    {
        it->second.tEndOk = Simulator::Now().GetSeconds();
        it->second.target = cellId;
    }
}

static void
OnUeHandoverEndError(uint64_t imsi, uint16_t, uint16_t)
{
    auto it = g_pendingHo.find(imsi);
    if (it != g_pendingHo.end())
    {
        EmitHo(imsi, it->second, "", "end_error");
        g_pendingHo.erase(it);
    }
}

static void
OnHandoverTotalTime(uint64_t imsi, uint16_t, uint16_t, Time totalTime)
{
    // Fired at the source gNB when the target releases the context (after path switch).
    PendingHo h;
    auto it = g_pendingHo.find(imsi);
    if (it != g_pendingHo.end())
    {
        h = it->second;
        g_pendingHo.erase(it);
    }
    EmitHo(imsi, h, Fmt(totalTime.GetSeconds() * 1000.0, 3), "ok");
}

static void
OnRadioLinkFailureCause(uint64_t imsi,
                        uint16_t cellId,
                        uint16_t,
                        uint16_t,
                        std::string cause,
                        int64_t,
                        int64_t)
{
    g_rlfOut << Fmt(Simulator::Now().GetSeconds(), 6) << "," << imsi << "," << cellId << ","
             << cause << "\n";
}

// ---------------------------------------------------------------------------------------------
// Actuation
// ---------------------------------------------------------------------------------------------

// Install a new m_priority for every class-`cls` flow the cell's scheduler currently serves.
static uint32_t
ApplyCellClassPriority(uint16_t cellId, const std::string& cls, uint8_t prio)
{
    auto it = g_cellSched.find(cellId);
    NS_ABORT_MSG_IF(it == g_cellSched.end(), "no scheduler for cell " << cellId);
    Ptr<C2QosScheduler> sched = it->second;
    uint32_t n = 0;
    for (auto& [rnti, info] : sched->m_registeredUes)
    {
        auto itIdx = g_imsiToIdx.find(sched->ImsiOf(rnti));
        if (itIdx != g_imsiToIdx.end() && g_ueClass[itIdx->second] == cls)
        {
            n += sched->SetFlowPriority(rnti, prio);
        }
    }
    return n;
}

static void
SetLevel(uint16_t cellId, const std::string& cls, uint8_t newPrio, const char* source)
{
    uint8_t oldPrio = g_currentPriority[cellId][cls];
    if (oldPrio == newPrio)
    {
        return;
    }
    g_currentPriority[cellId][cls] = newPrio;
    ApplyCellClassPriority(cellId, cls, newPrio);
    g_enforcementOut << Fmt(Simulator::Now().GetSeconds(), 6) << "," << cellId << "," << cls
                     << "," << +oldPrio << "," << +newPrio << "," << source << "\n";
}

// E2 control install from the enforcement schedule: new base level for `cls` on every cell in
// scope at once; the xApp offset is reset to the new base.
static void
ApplyPolicyUpdate(std::string scope, std::string cls, uint8_t newBase)
{
    for (uint16_t cellId : g_clusters.at(scope))
    {
        g_basePriority[cellId][cls] = newBase;
        SetLevel(cellId, cls, newBase, "policy");
    }
}

static void
ResetControlWindow()
{
    for (auto& [cellId, clsMap] : g_xappBuckets)
    {
        for (auto& [cls, bucket] : clsMap)
        {
            bucket.Reset();
        }
    }
    std::fill(g_xappUeTxPkts.begin(), g_xappUeTxPkts.end(), 0);
    g_rq5DlSymbols.clear();
    g_rq5CellAvailSymbols.clear();
}

// Numerical xApp, every 1 s: per cell and class, read the last-1 s counters of the UEs the cell
// serves, move m_priority by one step toward the target, clamp to base +/- band.
static void
RunNumericalXapp(Time interval)
{
    if (g_xappEnabled)
    {
        for (auto& [cellId, clsMap] : g_currentPriority)
        {
            for (const auto& cls : ALL_CLASSES)
            {
                auto itT = g_classTargets.find(cls);
                if (itT == g_classTargets.end())
                {
                    continue; // no target
                }
                auto itCell = g_xappBuckets.find(cellId);
                if (itCell == g_xappBuckets.end())
                {
                    continue;
                }
                auto itBucket = itCell->second.find(cls);
                if (itBucket == itCell->second.end())
                {
                    continue;
                }
                auto& bucket = itBucket->second;
                // Served UEs, each weighted by the share of its window sends made while this
                // cell served it (a UE that hands over or hits RLF mid-window counts partially).
                double nUes = 0;
                uint32_t txPkts = 0;
                for (const auto& [idx, n] : bucket.txPktsByUe)
                {
                    nUes += static_cast<double>(n) / g_xappUeTxPkts[idx];
                    txPkts += n;
                }
                if (txPkts == 0)
                {
                    continue; // no UE of this class served by the cell in the window: no reading
                }

                const ClassTarget& tgt = itT->second;
                int cur = clsMap[cls];
                int desired = cur;
                if (tgt.targetType == "throughput_floor_kbps")
                {
                    double sumKbps = bucket.rxBytes * 8.0 / interval.GetSeconds() / 1000.0;
                    double kbps = sumKbps / nUes; // mean per-UE throughput while served
                    if (kbps < tgt.targetValue)
                    {
                        desired = cur - g_xappStep; // miss: lower m_priority = higher weight
                    }
                    else if (kbps >= tgt.targetValue * (1.0 + g_xappMargin))
                    {
                        desired = cur + g_xappStep;
                    }
                }
                else if (tgt.targetType == "delay_p95_ms")
                {
                    if (bucket.delaysMs.empty())
                    {
                        desired = cur - g_xappStep; // packets sent, none delivered: miss
                    }
                    else
                    {
                        double p95 = P95(bucket.delaysMs);
                        if (p95 > tgt.targetValue)
                        {
                            desired = cur - g_xappStep;
                        }
                        else if (p95 <= tgt.targetValue * (1.0 - g_xappMargin))
                        {
                            desired = cur + g_xappStep;
                        }
                    }
                }
                else
                {
                    NS_FATAL_ERROR("unknown target_type " << tgt.targetType);
                }
                int base = g_basePriority[cellId][cls];
                int lo = std::max(1, base - g_xappBand);
                int hi = std::min(99, base + g_xappBand);
                SetLevel(cellId, cls, static_cast<uint8_t>(std::clamp(desired, lo, hi)), "xapp");
            }
        }
    }

    ResetControlWindow();
    Simulator::Schedule(interval, &RunNumericalXapp, interval);
}

static void
SendAll(const std::string& data)
{
    size_t sent = 0;
    while (sent < data.size())
    {
        ssize_t n = send(g_rq5Socket, data.data() + sent, data.size() - sent, 0);
        NS_ABORT_MSG_IF(n <= 0, "RQ5 socket send failed");
        sent += static_cast<size_t>(n);
    }
}

static json
Rq5Exchange(const json& request)
{
    SendAll(request.dump() + "\n");
    std::string line;
    char ch = 0;
    while (true)
    {
        ssize_t n = recv(g_rq5Socket, &ch, 1, 0);
        NS_ABORT_MSG_IF(n <= 0, "RQ5 controller closed the socket");
        if (ch == '\n')
        {
            break;
        }
        line.push_back(ch);
        NS_ABORT_MSG_IF(line.size() > 16 * 1024 * 1024, "RQ5 response exceeds 16 MiB");
    }
    try
    {
        return json::parse(line);
    }
    catch (const std::exception& e)
    {
        NS_FATAL_ERROR("invalid RQ5 controller JSON: " << e.what());
    }
    return json();
}

struct Rq5Action
{
    uint16_t cellId;
    std::string cls;
    std::string action;
};

static void
ApplyInterpreterDecision(std::vector<Rq5Action> actions)
{
    for (const auto& action : actions)
    {
        int current = g_currentPriority[action.cellId][action.cls];
        int desired = current;
        if (action.action == "raise")
        {
            desired -= g_xappStep; // lower m_priority = higher scheduler weight
        }
        else if (action.action == "lower")
        {
            desired += g_xappStep;
        }
        int base = g_basePriority[action.cellId][action.cls];
        int lo = std::max(1, base - g_xappBand);
        int hi = std::min(99, base + g_xappBand);
        SetLevel(action.cellId,
                 action.cls,
                 static_cast<uint8_t>(std::clamp(desired, lo, hi)),
                 "interpreter");
    }
}

// RQ5 arm (b), every 1 s. The blocking socket exchange freezes simulated time while the
// controller measures the live interpreter latency. A returned action is applied after that
// latency plus the frozen 5 ms E2 delay. A still-pending decision makes the next tick a logged skip.
static void
RunRq5Interpreter(Time interval)
{
    const double now = Simulator::Now().GetSeconds();
    if (g_rq5PendingUntil > now + 1e-12)
    {
        json skipped = {{"type", "skipped"},
                        {"tick_s", now},
                        {"pending_until_s", g_rq5PendingUntil}};
        json ack = Rq5Exchange(skipped);
        NS_ABORT_MSG_IF(ack.value("type", "") != "ack", "RQ5 skip was not acknowledged");
        ResetControlWindow();
        Simulator::Schedule(interval, &RunRq5Interpreter, interval);
        return;
    }

    json rows = json::array();
    for (const auto& [cellId, clsMap] : g_currentPriority)
    {
        uint64_t availableSymbols = g_rq5CellAvailSymbols[cellId];
        for (const auto& [cls, target] : g_classTargets)
        {
            auto& bucket = g_xappBuckets[cellId][cls];
            double nUes = 0.0;
            uint32_t txPkts = 0;
            for (const auto& [idx, n] : bucket.txPktsByUe)
            {
                nUes += static_cast<double>(n) / g_xappUeTxPkts[idx];
                txPkts += n;
            }
            json throughput = nullptr;
            if (txPkts > 0 && nUes > 0)
            {
                throughput = bucket.rxBytes * 8.0 / interval.GetSeconds() / 1000.0 / nUes;
            }
            json p95 = nullptr;
            if (!bucket.delaysMs.empty())
            {
                p95 = P95(bucket.delaysMs);
            }
            rows.push_back({{"cell", cellId},
                            {"class", cls},
                            {"throughput_kbps", throughput},
                            {"p95_delay_ms", p95},
                            {"prb_share",
                             availableSymbols > 0
                                 ? static_cast<double>(g_rq5DlSymbols[cellId][cls]) / availableSymbols
                                 : 0.0},
                            {"base_priority", g_basePriority[cellId][cls]},
                            {"current_priority", clsMap.at(cls)},
                            {"target_type", target.targetType},
                            {"target_value", target.targetValue}});
        }
    }
    json request = {{"type", "snapshot"},
                    {"tick_s", now},
                    {"window_s", interval.GetSeconds()},
                    {"step", g_xappStep},
                    {"band", g_xappBand},
                    {"rows", rows}};
    json response = Rq5Exchange(request);
    NS_ABORT_MSG_IF(response.value("type", "") != "decision", "RQ5 response is not a decision");
    NS_ABORT_MSG_IF(std::abs(response.at("tick_s").get<double>() - now) > 1e-9,
                    "RQ5 response tick mismatch");
    double latency = response.at("latency_s").get<double>();
    NS_ABORT_MSG_IF(!std::isfinite(latency) || latency < 0, "RQ5 response has invalid latency");

    std::map<std::pair<uint16_t, std::string>, std::string> returned;
    for (const auto& item : response.at("actions"))
    {
        uint16_t cellId = item.value("cell", 0);
        std::string cls = item.value("class", "");
        std::string action = item.value("action", "hold");
        if (g_currentPriority.count(cellId) &&
            std::find(ALL_CLASSES.begin(), ALL_CLASSES.end(), cls) != ALL_CLASSES.end() &&
            (action == "raise" || action == "hold" || action == "lower"))
        {
            returned[{cellId, cls}] = action;
        }
    }
    std::vector<Rq5Action> actions;
    for (const auto& [cellId, clsMap] : g_currentPriority)
    {
        for (const auto& targetEntry : g_classTargets)
        {
            const std::string& cls = targetEntry.first;
            auto it = returned.find({cellId, cls});
            actions.push_back({cellId, cls, it == returned.end() ? "hold" : it->second});
        }
    }
    g_rq5PendingUntil = now + latency + RQ5_DELTA_E2_S;
    Simulator::Schedule(Seconds(latency + RQ5_DELTA_E2_S), &ApplyInterpreterDecision, actions);
    ResetControlWindow();
    Simulator::Schedule(interval, &RunRq5Interpreter, interval);
}

// 100 ms reporter.
static void
ReportSlots(Time interval)
{
    std::string t = Fmt(Simulator::Now().GetSeconds(), 3);

    for (uint32_t i = 0; i < g_ueDevs.size(); ++i)
    {
        auto& s = g_sinks[i];
        g_ueSlotsOut << t << "," << g_ueDevs[i]->GetImsi() << ","
                     << g_ueDevs[i]->GetRrc()->GetCellId() << "," << g_ueClass[i] << ","
                     << s->m_slotRxBytes << "," << s->m_slotRxPkts << ",";
        if (s->m_slotRxPkts > 0)
        {
            double sum = 0;
            for (double d : s->m_slotDelaysMs)
            {
                sum += d;
            }
            double mean = sum / s->m_slotDelaysMs.size();
            g_ueSlotsOut << Fmt(mean, 3) << "," << Fmt(P95(s->m_slotDelaysMs), 3) << "\n";
        }
        else
        {
            g_ueSlotsOut << ",\n";
        }
        s->m_slotRxBytes = 0;
        s->m_slotRxPkts = 0;
        s->m_slotDelaysMs.clear();

        auto& f = g_source->m_flows[i];
        g_txTraceOut << t << "," << f.imsi << "," << f.slotTxBytes << "," << f.slotTxPkts << "\n";
        f.slotTxBytes = 0;
        f.slotTxPkts = 0;

        Vector pos = g_ueNodes[i]->GetObject<MobilityModel>()->GetPosition();
        g_positionsOut << t << "," << g_ueDevs[i]->GetImsi() << "," << Fmt(pos.x, 3) << ","
                       << Fmt(pos.y, 3) << "," << g_ueDevs[i]->GetRrc()->GetCellId() << "\n";
    }

    for (auto& [cellId, dev] : g_cellGnbDevs)
    {
        uint64_t used = g_cellUsedReg[cellId];
        uint64_t avail = g_cellAvailReg[cellId];
        g_cellPrbOut << t << "," << cellId << ","
                     << Fmt(avail > 0 ? static_cast<double>(used) / avail : 0.0, 4) << "\n";
        g_cellUsedReg[cellId] = 0;
        g_cellAvailReg[cellId] = 0;

        Ptr<C2QosScheduler> sched = g_cellSched[cellId];
        for (auto& [rnti, info] : sched->m_registeredUes)
        {
            uint64_t imsi = sched->ImsiOf(rnti);
            auto itIdx = g_imsiToIdx.find(imsi);
            int p = sched->GetFlowPriority(rnti);
            bool known = itIdx != g_imsiToIdx.end();
            std::string cls = known ? g_ueClass[itIdx->second] : "";
            g_lcPrioOut << t << "," << cellId << "," << rnti << "," << imsi << "," << cls << ","
                        << (p >= 0 ? std::to_string(p) : "") << ","
                        << (known ? std::to_string(g_currentPriority[cellId][cls]) : "") << "\n";
        }
    }

    Simulator::Schedule(interval, &ReportSlots, interval);
}

// ---------------------------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
    std::string configPath = "c2-default.json";
    std::string schedulePath;
    std::string rq5SocketPath;
    std::string outputDir = ".";
    uint32_t rngRun = 1;
    double simTimeOverride = -1.0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("config", "Path to JSON scenario configuration", configPath);
    cmd.AddValue("schedule", "Path to CSV enforcement schedule (t_s,scope,class,priority)", schedulePath);
    cmd.AddValue("rq5Socket", "Unix socket for the RQ5 arm-(b) controller (empty disables the hook)", rq5SocketPath);
    cmd.AddValue("outputDir", "Output directory", outputDir);
    cmd.AddValue("rngRun", "RngRun (RngSeed comes from the config)", rngRun);
    cmd.AddValue("simTime", "Simulated time override in seconds", simTimeOverride);
    cmd.Parse(argc, argv);

    std::ifstream configFile(configPath);
    NS_ABORT_MSG_IF(!configFile.is_open(), "cannot open config " << configPath);
    json cfg;
    configFile >> cfg;

    const json& sc = cfg.at("scenario");
    const json& ue = cfg.at("ues");
    const uint32_t numCells = sc.at("num_cells").get<uint32_t>();
    NS_ABORT_MSG_IF(numCells != 1 && numCells != 21, "num_cells must be 1 or 21");
    const std::string scenarioName = sc.at("name").get<std::string>();
    const double isdM = sc.at("isd_m").get<double>();
    const double bsHeightM = sc.at("bs_height_m").get<double>();
    const double utHeightM = sc.at("ut_height_m").get<double>();
    const double minBsUtDistM = sc.at("min_bs_ut_distance_m").get<double>();
    const double carrierHz = sc.at("carrier_freq_hz").get<double>();
    const double bandwidthMhz = sc.at("bandwidth_mhz").get<double>();
    const uint16_t numerology = sc.at("numerology").get<uint16_t>();
    const double gnbTxDbm = sc.at("gnb_tx_power_dbm").get<double>();
    const double ueTxDbm = sc.at("ue_tx_power_dbm").get<double>();
    const double gnbNf = sc.at("gnb_noise_figure_db").get<double>();
    const double ueNf = sc.at("ue_noise_figure_db").get<double>();
    const uint32_t gnbRows = sc.at("gnb_antenna_rows").get<uint32_t>();
    const uint32_t gnbCols = sc.at("gnb_antenna_cols").get<uint32_t>();
    const double gnbHSp = sc.at("gnb_h_spacing").get<double>();
    const double gnbVSp = sc.at("gnb_v_spacing").get<double>();
    const double downtiltDeg = sc.at("gnb_downtilt_deg").get<double>();
    const double bfPeriodMs = sc.at("beamforming_periodicity_ms").get<double>();
    const bool enableShadowing = sc.at("enable_shadowing").get<bool>();
    const bool enableFading = sc.at("enable_fading").get<bool>();
    const uint32_t rlcBufBytes = sc.at("rlc_max_tx_buffer_bytes").get<uint32_t>();
    const std::string tddPattern = sc.at("tdd_pattern").get<std::string>();

    const uint32_t uesPerCell = ue.at("ues_per_cell").get<uint32_t>();
    const double speedKmh = ue.at("speed_kmh").get<double>();
    const double hysteresisDb = ue.at("a3_hysteresis_db").get<double>();
    const uint32_t tttMs = ue.at("a3_time_to_trigger_ms").get<uint32_t>();
    const uint32_t hoTrigDelayMs = ue.at("handover_triggering_delay_ms").get<uint32_t>();
    const uint32_t hoJoinMs = ue.at("handover_joining_timeout_ms").get<uint32_t>();
    const std::string initialAttach = ue.at("initial_attach").get<std::string>();
    NS_ABORT_MSG_IF(initialAttach != "closest" && initialAttach != "max_rsrp",
                    "initial_attach must be closest or max_rsrp");

    const json& tr = cfg.at("traffic");
    const double trafficStart = tr.at("start_time_s").get<double>();
    const double startJitterMs = tr.at("start_jitter_ms").get<double>();

    double simTimeSec = cfg.at("simulation").at("sim_time_s").get<double>();
    if (simTimeOverride > 0)
    {
        simTimeSec = simTimeOverride;
    }
    const Time simTime = Seconds(simTimeSec);
    const uint32_t rngSeed = cfg.at("simulation").at("rng_seed").get<uint32_t>();

    for (auto& [lvl, val] : cfg.at("priorities").at("levels").items())
    {
        g_priorityLevels[lvl] = val.get<uint8_t>();
    }
    for (const char* lvl : {"critical", "high", "normal", "low"})
    {
        NS_ABORT_MSG_IF(!g_priorityLevels.count(lvl), "missing priority level " << lvl);
    }
    g_defaultPriority = g_priorityLevels.at(cfg.at("priorities").at("default_policy").get<std::string>());

    for (auto& [name, cells] : cfg.at("clusters").items())
    {
        if (name.rfind("_", 0) == 0)
        {
            continue; // documentation keys
        }
        g_clusters[name] = cells.get<std::vector<uint16_t>>();
        for (uint16_t c : g_clusters[name])
        {
            NS_ABORT_MSG_IF(c < 1 || c > numCells, "cluster " << name << " has bad cell " << c);
        }
    }

    std::vector<std::string> ueClasses = ALL_CLASSES;
    if (cfg.contains("ue_classes"))
    {
        ueClasses = cfg.at("ue_classes").get<std::vector<std::string>>();
    }
    for (auto& [cls, obj] : tr.at("classes").items())
    {
        NS_ABORT_MSG_IF(std::find(ALL_CLASSES.begin(), ALL_CLASSES.end(), cls) == ALL_CLASSES.end(),
                        "unknown class " << cls);
        ClassTarget t;
        t.type = obj.at("type").get<std::string>();
        NS_ABORT_MSG_IF(t.type != "cbr" && t.type != "periodic" && t.type != "backlogged",
                        "class " << cls << ": traffic type must be cbr, periodic or backlogged");
        t.rateKbps = obj.at("rate_kbps").get<double>();
        t.packetSize = obj.at("packet_size").get<uint32_t>();
        t.targetType = obj.at("target_type").get<std::string>();
        t.targetValue = obj.at("target_value").get<double>();
        g_classTargets[cls] = t;
    }
    for (const auto& c : ueClasses)
    {
        NS_ABORT_MSG_IF(!g_classTargets.count(c), "class " << c << " has no traffic entry");
    }

    const json& xa = cfg.at("xapp");
    g_xappEnabled = xa.at("enabled").get<bool>();
    const double xappInterval = xa.at("interval_s").get<double>();
    g_xappStep = xa.at("step").get<uint8_t>();
    g_xappMargin = xa.at("margin").get<double>();
    g_xappBand = xa.at("band").get<uint8_t>();
    g_rq5Enabled = !rq5SocketPath.empty();
    NS_ABORT_MSG_IF(!rq5SocketPath.empty() && std::abs(xappInterval - 1.0) > 1e-12,
                    "RQ5 requires the frozen 1 s controller interval");

    RngSeedManager::SetSeed(rngSeed);
    RngSeedManager::SetRun(rngRun);

    for (uint16_t c = 1; c <= numCells; ++c)
    {
        for (const auto& cls : ALL_CLASSES)
        {
            g_basePriority[c][cls] = g_defaultPriority;
            g_currentPriority[c][cls] = g_defaultPriority;
        }
    }

    // Parse the schedule before building anything so a malformed file fails fast.
    struct SchedRow
    {
        double t;
        std::string scope, cls;
        uint8_t prio;
    };

    std::vector<SchedRow> schedRows;
    if (!schedulePath.empty())
    {
        std::ifstream sf(schedulePath);
        NS_ABORT_MSG_IF(!sf.is_open(), "cannot open schedule " << schedulePath);
        std::string line;
        bool header = true;
        while (std::getline(sf, line))
        {
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            if (line.find_first_not_of(" \t") == std::string::npos)
            {
                continue;
            }
            if (header)
            {
                NS_ABORT_MSG_IF(line != "t_s,scope,class,priority", "bad schedule header: " << line);
                header = false;
                continue;
            }
            std::stringstream ss(line);
            std::string ts, scope, cls, prio;
            std::getline(ss, ts, ',');
            std::getline(ss, scope, ',');
            std::getline(ss, cls, ',');
            std::getline(ss, prio, ',');
            NS_ABORT_MSG_IF(!g_clusters.count(scope), "schedule: unknown scope " << scope);
            NS_ABORT_MSG_IF(std::find(ALL_CLASSES.begin(), ALL_CLASSES.end(), cls) == ALL_CLASSES.end(),
                            "schedule: unknown class " << cls);
            uint8_t p = g_defaultPriority;
            if (prio != "default")
            {
                NS_ABORT_MSG_IF(!g_priorityLevels.count(prio), "schedule: unknown priority " << prio);
                p = g_priorityLevels.at(prio);
            }
            size_t consumed = 0;
            double t = 0.0;
            bool parseOk = !ts.empty() && !std::isspace(static_cast<unsigned char>(ts[0]));
            try
            {
                t = std::stod(ts, &consumed);
                if (consumed != ts.size() || !std::isfinite(t))
                {
                    parseOk = false;
                }
            }
            catch (...)
            {
                parseOk = false;
            }
            NS_ABORT_MSG_IF(!parseOk,
                            "schedule: invalid t_s token '" << ts << "' in line: " << line);
            NS_ABORT_MSG_IF(t < 0 || t > simTimeSec,
                            "schedule: t_s out of range " << t << " in line: " << line);
            schedRows.push_back({t, scope, cls, p});
        }
    }

    auto open = [&](std::ofstream& f, const char* name, const char* hdr) {
        f.open(outputDir + "/" + name, std::ios::out | std::ios::trunc);
        NS_ABORT_MSG_IF(!f.is_open(), "cannot write " << outputDir << "/" << name);
        f << hdr << "\n";
    };
    open(g_ueSlotsOut, "ue_slots.csv",
         "t_slot,imsi,serving_cell,class,rx_bytes,n_rx_pkts,mean_delay_ms,p95_delay_ms");
    open(g_txTraceOut, "tx_trace.csv", "t_slot,imsi,tx_bytes,n_tx_pkts");
    open(g_positionsOut, "positions.csv", "t,imsi,x,y,serving_cell");
    open(g_cellPrbOut, "cell_prb.csv", "t_slot,cell,prb_share");
    open(g_handoverOut, "handover.csv",
         "t_start,t_end_ok,imsi,source_cell,target_cell,handover_total_time_ms,outcome");
    open(g_rlfOut, "rlf.csv", "t,imsi,cell,cause");
    open(g_enforcementOut, "enforcement.csv", "t,cell,class,old,new,source");
    open(g_lcPrioOut, "lc_priority.csv", "t,cell,rnti,imsi,class,lc_m_priority,cell_class_level");

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(rlcBufBytes));
    Config::SetDefault("ns3::NrGnbRrc::HandoverTriggeringDelay", TimeValue(MilliSeconds(hoTrigDelayMs)));
    Config::SetDefault("ns3::NrGnbRrc::HandoverJoiningTimeoutDuration", TimeValue(MilliSeconds(hoJoinMs)));
    Config::SetDefault("ns3::RandomDirection2dMobilityModel::Pause",
                       StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue(MilliSeconds(0)));

    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> bfHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetBeamformingHelper(bfHelper);
    nrHelper->SetEpcHelper(epcHelper);
    epcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));
    bfHelper->SetAttribute("BeamformingMethod", TypeIdValue(CellScanBeamforming::GetTypeId()));
    bfHelper->SetAttribute("BeamformingPeriodicity", TimeValue(MilliSeconds(bfPeriodMs)));

    nrHelper->SetAttribute("UseIdealRrc", BooleanValue(false));
    nrHelper->SetSchedulerTypeId(C2QosScheduler::GetTypeId());
    nrHelper->SetSchedulerAttribute("EnableSrsInUlSlots", BooleanValue(false));
    nrHelper->SetSchedulerAttribute("EnableSrsInFSlots", BooleanValue(false));
    nrHelper->SetHandoverAlgorithmType("ns3::NrA3RsrpHandoverAlgorithm");
    nrHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(hysteresisDb));
    nrHelper->SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(MilliSeconds(tttMs)));

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories(scenarioName, "Default", "ThreeGpp");
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(enableShadowing));
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));

    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(carrierHz, bandwidthMhz * 1e6, 1);
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);
    channelHelper->AssignChannelsToBands(
        {band},
        enableFading ? (NrChannelHelper::INIT_PROPAGATION | NrChannelHelper::INIT_FADING)
                     : NrChannelHelper::INIT_PROPAGATION);
    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    NodeContainer gnbNodes;
    NodeContainer ueNodes;
    std::map<std::string, std::pair<int64_t, int64_t>> streams; // component -> [first, last]
    std::vector<uint16_t> dropCell;                             // by UE index
    std::string boxStr = "";
    HexagonalGridScenarioHelper grid;

    if (numCells == 1)
    {
        gnbNodes.Create(1);
        ueNodes.Create(uesPerCell);
        MobilityHelper mob;
        mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        Ptr<ListPositionAllocator> gp = CreateObject<ListPositionAllocator>();
        gp->Add(Vector(0.0, 0.0, bsHeightM));
        mob.SetPositionAllocator(gp);
        mob.Install(gnbNodes);
        // Spike-A geometry: UEs symmetric at 10 m (static; no mobility RNG).
        Ptr<ListPositionAllocator> up = CreateObject<ListPositionAllocator>();
        for (uint32_t u = 0; u < uesPerCell; ++u)
        {
            double y = (uesPerCell == 1) ? 0.0 : -2.0 + 4.0 * u / (uesPerCell - 1);
            up->Add(Vector(10.0, y, utHeightM));
            dropCell.push_back(1);
        }
        mob.SetPositionAllocator(up);
        mob.Install(ueNodes);
    }
    else
    {
        grid.SetSectorization(HexagonalGridScenarioHelper::TRIPLE);
        grid.SetNumRings(1);
        grid.m_isd = isdM;
        grid.m_bsHeight = bsHeightM;
        grid.m_utHeight = utHeightM;
        grid.m_minBsUtDistance = minBsUtDistM;
        grid.SetUtNumber(numCells * uesPerCell);
        grid.SetMaxUeDistanceToClosestSite(isdM);
        // Drop RNG streams must be pinned BEFORE the drop is drawn.
        int64_t n = grid.AssignStreams(STREAM_MOBILITY);
        streams["mobility_drop"] = {STREAM_MOBILITY, STREAM_MOBILITY + n - 1};
        double speedMs = speedKmh / 3.6;
        grid.CreateScenarioWithMobility(Vector(speedMs, 0, 0),
                                        Vector(speedMs, 0, 0),
                                        0.0,
                                        "ns3::RandomDirection2dMobilityModel");
        gnbNodes = grid.GetBaseStations();
        ueNodes = grid.GetUserTerminals();
        MobilityHelper mh;
        int64_t m = mh.AssignStreams(ueNodes, STREAM_MOBILITY + 100);
        streams["mobility_models"] = {STREAM_MOBILITY + 100, STREAM_MOBILITY + 100 + m - 1};
        // The helper drops UE utId in cell (utId % numCells) + 1 (round-robin over sectors).
        for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
        {
            dropCell.push_back(static_cast<uint16_t>(i % numCells) + 1);
        }
        RectangleValue bv;
        ueNodes.Get(0)->GetObject<RandomDirection2dMobilityModel>()->GetAttribute("Bounds", bv);
        Rectangle box = bv.Get();
        boxStr = "[" + Fmt(box.xMin, 1) + "," + Fmt(box.xMax, 1) + "," + Fmt(box.yMin, 1) + "," +
                 Fmt(box.yMax, 1) + "]";
    }
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        g_ueNodes.push_back(ueNodes.Get(i));
    }

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()));
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(gnbRows));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(gnbCols));
    nrHelper->SetGnbAntennaAttribute("AntennaHorizontalSpacing", DoubleValue(gnbHSp));
    nrHelper->SetGnbAntennaAttribute("AntennaVerticalSpacing", DoubleValue(gnbVSp));
    nrHelper->SetGnbAntennaAttribute("DowntiltAngle", DoubleValue(downtiltDeg * M_PI / 180.0));
    nrHelper->SetGnbAntennaAttribute("AntennaElement", PointerValue(CreateObject<ThreeGppAntennaModel>()));
    nrHelper->SetGnbPhyAttribute("NoiseFigure", DoubleValue(gnbNf));
    nrHelper->SetUePhyAttribute("NoiseFigure", DoubleValue(ueNf));
    nrHelper->SetUePhyAttribute("TxPower", DoubleValue(ueTxDbm));

    NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueDevs = nrHelper->InstallUeDevice(ueNodes, allBwps);

    std::ofstream cellsOut(outputDir + "/cells.csv");
    cellsOut << "cell,site,sector,x,y,bearing_deg\n";
    for (uint32_t i = 0; i < gnbDevs.GetN(); ++i)
    {
        Ptr<NrGnbNetDevice> dev = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(i));
        uint16_t cellId = dev->GetCellId();
        g_cellGnbDevs[cellId] = dev;
        uint32_t sector = (numCells == 1) ? 0 : (i % 3);
        double bearingDeg = 30.0 + 120.0 * sector;
        double bearing = bearingDeg * M_PI / 180.0;
        if (bearing > M_PI)
        {
            bearing -= 2.0 * M_PI;
        }
        Ptr<NrGnbPhy> phy = NrHelper::GetGnbPhy(dev, 0);
        ConstCast<UniformPlanarArray>(
            phy->GetSpectrumPhy()->GetAntenna()->GetObject<UniformPlanarArray>())
            ->SetAttribute("BearingAngle", DoubleValue(bearing));
        phy->SetAttribute("Numerology", UintegerValue(numerology));
        phy->SetAttribute("TxPower", DoubleValue(gnbTxDbm));
        phy->SetAttribute("Pattern", StringValue(tddPattern));
        phy->TraceConnectWithoutContext("SlotDataStats", MakeCallback(&OnPhySlotDataStats));
        if (!rq5SocketPath.empty())
        {
            NrHelper::GetGnbMac(dev, 0)->TraceConnectWithoutContext(
                "DlScheduling", MakeBoundCallback(&OnDlScheduling, cellId));
        }

        Ptr<C2QosScheduler> sched = DynamicCast<C2QosScheduler>(dev->GetScheduler(0));
        NS_ABORT_MSG_IF(!sched, "scheduler of cell " << cellId << " is not C2QosScheduler");
        sched->m_cellId = cellId;
        sched->m_rrc = PeekPointer(dev->GetRrc());
        sched->m_lookup = &LookupPriority;
        g_cellSched[cellId] = sched;

        Vector p = gnbNodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
        cellsOut << cellId << "," << (numCells == 1 ? 0 : i / 3) << "," << sector << ","
                 << Fmt(p.x, 2) << "," << Fmt(p.y, 2) << "," << Fmt(bearingDeg, 0) << "\n";
    }
    cellsOut.close();

    auto [remoteHost, remoteHostAddr] = epcHelper->SetupRemoteHost("100Gb/s", 2500, Seconds(0.0));
    InternetStackHelper internet;
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIp = epcHelper->AssignUeIpv4Address(ueDevs);

    // RNG streams: "everything else" blocks via the NR helper, then the channel objects are
    // re-pinned to their own block (the helper would otherwise place them inside the gNB block).
    {
        // Count each block separately so overlap can be asserted.
        int64_t nEpc = epcHelper->AssignStreams(STREAM_EPC);
        int64_t nRh = internet.AssignStreams(NodeContainer(remoteHost), STREAM_REMOTE);
        int64_t nUeSt = internet.AssignStreams(ueNodes, STREAM_UE_STACK);
        int64_t nGnb = nrHelper->AssignStreams(gnbDevs, STREAM_GNB_DEV);
        int64_t nUe = nrHelper->AssignStreams(ueDevs, STREAM_UE_DEV);
        streams["other_epc"] = {STREAM_EPC, STREAM_EPC + nEpc - 1};
        streams["other_remote_host_stack"] = {STREAM_REMOTE, STREAM_REMOTE + nRh - 1};
        streams["other_ue_stacks"] = {STREAM_UE_STACK, STREAM_UE_STACK + nUeSt - 1};
        streams["other_gnb_devices"] = {STREAM_GNB_DEV, STREAM_GNB_DEV + nGnb - 1};
        streams["other_ue_devices"] = {STREAM_UE_DEV, STREAM_UE_DEV + nUe - 1};

        Ptr<SpectrumChannel> ch = NrHelper::GetGnbPhy(gnbDevs.Get(0), 0)->GetSpectrumPhy()->GetSpectrumChannel();
        int64_t s = STREAM_CHANNEL;
        Ptr<ThreeGppPropagationLossModel> pl =
            DynamicCast<ThreeGppPropagationLossModel>(ch->GetPropagationLossModel());
        NS_ABORT_MSG_IF(!pl, "expected a ThreeGppPropagationLossModel");
        s += pl->AssignStreams(s);
        s += pl->GetChannelConditionModel()->AssignStreams(s);
        Ptr<ThreeGppSpectrumPropagationLossModel> sl =
            DynamicCast<ThreeGppSpectrumPropagationLossModel>(ch->GetPhasedArraySpectrumPropagationLossModel());
        NS_ABORT_MSG_IF(enableFading && !sl, "fading enabled but no 3GPP spectrum model");
        if (sl)
        {
            s += DynamicCast<ThreeGppChannelModel>(sl->GetChannelModel())->AssignStreams(s);
        }
        streams["channel"] = {STREAM_CHANNEL, s - 1};
    }

    if (numCells == 1)
    {
        for (uint32_t i = 0; i < ueDevs.GetN(); ++i)
        {
            nrHelper->AttachToGnb(ueDevs.Get(i), gnbDevs.Get(0));
        }
    }
    else
    {
        if (initialAttach == "max_rsrp")
        {
            // RSRP-based association draws from the helper's per-UE stream block
            // (NrHelper::INIT_ASSOC_STREAM_BASE + node id), outside the blocks above.
            nrHelper->AttachToMaxRsrpGnb(ueDevs, gnbDevs);
        }
        else
        {
            nrHelper->AttachToClosestGnb(ueDevs, gnbDevs);
        }
        nrHelper->AddX2Interface(gnbNodes);
    }

    Ptr<UniformRandomVariable> jitterRng = CreateObject<UniformRandomVariable>();
    jitterRng->SetStream(STREAM_TRAFFIC);
    streams["traffic"] = {STREAM_TRAFFIC, STREAM_TRAFFIC};

    g_source = CreateObject<C2TrafficSource>();
    remoteHost->AddApplication(g_source);

    std::ofstream ueMapOut(outputDir + "/ue_map.csv");
    ueMapOut << "ue_index,imsi,drop_cell,class,port\n";
    std::vector<uint32_t> perCellCount(numCells + 1, 0);
    g_xappUeTxPkts.assign(ueDevs.GetN(), 0);
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i)
    {
        Ptr<NrUeNetDevice> dev = DynamicCast<NrUeNetDevice>(ueDevs.Get(i));
        g_ueDevs.push_back(dev);
        // Round-robin within each cell: the k-th UE dropped in a cell gets class k mod |classes|.
        uint32_t k = perCellCount[dropCell[i]]++;
        std::string cls = ueClasses[k % ueClasses.size()];
        g_ueClass.push_back(cls);
        g_imsiToIdx[dev->GetImsi()] = i;
        uint16_t port = 12000 + i;

        Ptr<NrQosRule> rule = Create<NrQosRule>();
        NrQosRule::PacketFilter filter;
        filter.localPortStart = port;
        filter.localPortEnd = port;
        rule->Add(filter);
        nrHelper->ActivateDedicatedQosFlow(dev, NrQosFlow(NrQosFlow::NGBR_LOW_LAT_EMBB), rule);

        Ptr<C2TrafficSink> sink = CreateObject<C2TrafficSink>();
        sink->Setup(i, port);
        ueNodes.Get(i)->AddApplication(sink);
        sink->SetStartTime(Seconds(0.0));
        sink->SetStopTime(simTime);
        g_sinks.push_back(sink);

        const ClassTarget& t = g_classTargets.at(cls);
        UeFlowSpec spec;
        spec.imsi = dev->GetImsi();
        spec.destIp = ueIp.GetAddress(i);
        spec.destPort = port;
        spec.pktSize = t.packetSize;
        if (t.type == "backlogged")
        {
            // Full buffer: 8 bit on every resource element of the carrier. Subcarriers
            // B/(15 kHz 2^mu) x symbols 14 x 1000 x 2^mu per second = B x 14000/15000 RE/s for any
            // numerology. rate_kbps is not used.
            const double backloggedBps = 8.0 * bandwidthMhz * 1e6 * 14000.0 / 15000.0;
            spec.interval = Seconds(t.packetSize * 8.0 / backloggedBps);
        }
        else
        {
            // cbr and periodic: one packet every packet_size*8/rate_kbps seconds.
            spec.interval = Seconds(t.packetSize * 8.0 / (t.rateKbps * 1000.0));
        }
        spec.startTime = Seconds(trafficStart + jitterRng->GetValue(0.0, startJitterMs / 1000.0));
        g_source->m_flows.push_back(spec);

        ueMapOut << i << "," << dev->GetImsi() << "," << dropCell[i] << "," << cls << "," << port
                 << "\n";
    }
    ueMapOut.close();
    g_source->SetStartTime(Seconds(0.0));
    g_source->SetStopTime(simTime);

    // Stream blocks must not overlap.
    {
        std::vector<std::pair<int64_t, int64_t>> blocks;
        for (auto& [name, r] : streams)
        {
            if (r.second >= r.first)
            {
                blocks.push_back(r);
            }
        }
        std::sort(blocks.begin(), blocks.end());
        for (size_t i = 1; i < blocks.size(); ++i)
        {
            NS_ABORT_MSG_IF(blocks[i].first <= blocks[i - 1].second, "RNG stream blocks overlap");
        }
        NS_ABORT_MSG_IF(!blocks.empty() && blocks.back().second >= STREAM_BLOCK_END,
                        "RNG stream block beyond " << STREAM_BLOCK_END);
        std::ofstream so(outputDir + "/streams.json");
        json j;
        j["rng_seed"] = rngSeed;
        j["rng_run"] = rngRun;
        for (auto& [name, r] : streams)
        {
            j["blocks"][name] = {r.first, r.second};
        }
        j["ue_bounding_box"] = boxStr;
        so << j.dump(2) << "\n";
    }

    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/NrGnbRrc/HandoverStart",
                                  MakeCallback(&OnGnbHandoverStart));
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/NrUeRrc/HandoverEndOk",
                                  MakeCallback(&OnUeHandoverEndOk));
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/NrUeRrc/HandoverEndError",
                                  MakeCallback(&OnUeHandoverEndError));
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/NrGnbRrc/HandoverTotalTime",
                                  MakeCallback(&OnHandoverTotalTime));
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/NrUeRrc/RadioLinkFailureCause",
                                  MakeCallback(&OnRadioLinkFailureCause));

    if (!rq5SocketPath.empty())
    {
        sockaddr_un addr{};
        NS_ABORT_MSG_IF(rq5SocketPath.size() >= sizeof(addr.sun_path),
                        "RQ5 Unix socket path is too long");
        g_rq5Socket = socket(AF_UNIX, SOCK_STREAM, 0);
        NS_ABORT_MSG_IF(g_rq5Socket < 0, "cannot create RQ5 Unix socket");
        addr.sun_family = AF_UNIX;
        std::copy(rq5SocketPath.begin(), rq5SocketPath.end(), addr.sun_path);
        NS_ABORT_MSG_IF(connect(g_rq5Socket, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0,
                        "cannot connect RQ5 Unix socket " << rq5SocketPath);
    }

    for (const auto& r : schedRows)
    {
        Simulator::Schedule(Seconds(r.t), &ApplyPolicyUpdate, r.scope, r.cls, r.prio);
    }
    Simulator::Schedule(MilliSeconds(100), &ReportSlots, MilliSeconds(100));
    if (rq5SocketPath.empty())
    {
        Simulator::Schedule(Seconds(xappInterval), &RunNumericalXapp, Seconds(xappInterval));
    }
    else
    {
        Simulator::Schedule(Seconds(xappInterval), &RunRq5Interpreter, Seconds(xappInterval));
    }

    // Stop just after simTime so the reporter's window ending at simTime is written.
    Simulator::Stop(simTime + MicroSeconds(1));
    Simulator::Run();

    for (auto& [imsi, h] : g_pendingHo)
    {
        EmitHo(imsi, h, "", "incomplete");
    }
    Simulator::Destroy();

    if (g_rq5Socket >= 0)
    {
        close(g_rq5Socket);
        g_rq5Socket = -1;
    }

    for (auto* f : {&g_ueSlotsOut, &g_txTraceOut, &g_positionsOut, &g_cellPrbOut, &g_handoverOut,
                    &g_rlfOut, &g_enforcementOut, &g_lcPrioOut})
    {
        f->close();
    }
    return 0;
}
