//! mcp — the control MCP (stdio, JSON-RPC 2.0) an agent drives the
//! project through; this library holds its Jev wire (`jev`) and the events
//! tool (`events`), the `ast_grep` tool (`astgrep`) and the tool-call log and
//! its graph (`usage`); `main.rs` is the server itself.

pub mod astgrep;
pub mod audit;
pub mod batch;
pub mod docs;
pub mod events;
pub mod jev;
pub mod usage;
