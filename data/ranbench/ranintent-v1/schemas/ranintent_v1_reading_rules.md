# RANIntent v1 reading rules

Turn one natural-language RAN intent into one policy of A1 policy type 20001 (RANIntent v1).
Each case gives the issuer (the network operator, or a tenant with the traffic classes it owns and its rights),
the intent text, and a KPM telemetry table with one or more rows per radio cell; each row names the cell's cluster.

## Fields

Fill every field with exactly one allowed value.

### action
- `prioritise`: raise the scheduling priority of the traffic class
- `deprioritise`: lower the scheduling priority of the traffic class
- `guarantee_share`: reserve a minimum share of the radio resource blocks for the class
- `cap_share`: limit the class to a maximum share of the radio resource blocks
- `restrict_edge`: keep the class's edge processing no further out than a given edge location
- `relax_edge`: allow the class's edge processing to move out as far as a given edge location
- `set_latency_target`: set a packet delay target for the class
- `revert_default`: return the class to its default treatment, cancelling earlier changes
- `unsupported`: the request cannot be expressed with any of the actions above

### class
- `emergency_video`: live video from emergency services such as body cameras and ambulance cameras
- `mission_voice`: mission-critical push-to-talk voice of first responders
- `factory_control`: closed-loop industrial control traffic of automated machines
- `xr_gaming`: interactive extended-reality and cloud gaming sessions
- `video_streaming`: consumer video streaming
- `iot_metering`: low-rate meter and sensor reports
- `analytics_offload`: bulk uploads to edge analytics jobs
- `best_effort`: all other best-effort data

### scope
- `all_cells`: every cell of the network
- `north_cluster`: the north cluster
- `south_cluster`: the south cluster
- `stadium`: the stadium cluster
- `hospital_zone`: the hospital zone cluster
- `most_loaded_cells`: the most heavily loaded cells, identified from the telemetry
- `worst_edge_cells`: the cells whose cell-edge users get the worst throughput, identified from the telemetry
- `unspecified`: no location stated

### target_cluster
- `north_cluster`: the north cluster
- `south_cluster`: the south cluster
- `stadium`: the stadium cluster
- `hospital_zone`: the hospital zone cluster
- `none`: the scope is not state-dependent

### priority
- `critical`: critical, the highest priority level
- `high`: high priority
- `normal`: normal priority
- `low`: low priority
- `unspecified`: no priority level stated

### prb_share
- `10`: 10 percent of the resource blocks
- `20`: 20 percent of the resource blocks
- `30`: 30 percent of the resource blocks
- `50`: 50 percent of the resource blocks
- `70`: 70 percent of the resource blocks
- `unspecified`: no resource share stated

### edge_site
- `on_site`: on-site edge servers at the cell sites
- `metro_edge`: a metro edge data centre
- `regional_cloud`: a regional cloud data centre
- `any`: any edge or cloud location
- `unspecified`: no edge location stated

### latency_target
- `5`: 5 ms
- `10`: 10 ms
- `20`: 20 ms
- `50`: 50 ms
- `100`: 100 ms
- `unspecified`: no delay target stated

### duration
- `until_revoked`: until someone revokes it
- `15min`: for 15 minutes
- `1h`: for one hour
- `event_end`: until the current event ends
- `unspecified`: no duration stated

## Field rules

1. A field the intent does not state is `unspecified` (`target_cluster`: `none`). Do not fill in defaults.
2. `revert_default` carries no priority: `priority` is `unspecified`.
3. `unsupported`: the intent asks for something no action expresses. Then `scope`, `priority`, `prb_share`, `edge_site`, `latency_target` and `duration` are `unspecified` and `target_cluster` is `none`; `class` is the class the intent concerns.
4. `scope` is a named cluster when the intent names one, `all_cells` when it covers the whole network, and `most_loaded_cells` or `worst_edge_cells` when it identifies the cells only by their current state.
5. `target_cluster` is set only for the two state-dependent scopes: for `most_loaded_cells` it is the cluster that contains the cell with the highest `prb_util_pct`; for `worst_edge_cells` it is the cluster that contains the cell with the lowest `edge_ue_thr_mbps` (the throughput of the cell's cell-edge users). For every other scope it is `none`.
6. A tenant acts only on the classes it owns and may set `critical` priority only if it holds that right; the operator owns every class and may set `critical`.

## Telemetry rules

- Fresh: when rows carry neither `measured_at` nor `source`, the table is the state.
- Stale: when rows carry `measured_at`, rows older than 10 s (relative to the snapshot time in the table header) are ignored in favour of the newest row per cell.
- Noisy: when the header states noise bounds, values carry bounded noise that never changes which cells satisfy the scope; use the values as given.
- Contradictory: when rows carry `source`, E2 KPM rows (`E2_KPM`) override O1 PM rows (`O1_PM`) for the same cell and metric.

## Output

One JSON object with exactly the nine fields above, each set to one allowed value.
