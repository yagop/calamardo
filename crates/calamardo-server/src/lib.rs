//! OpenAI-compatible server for Inkling 2 Mini (milestone 4).

pub mod metrics;
pub mod parser;
pub mod server;
pub mod template;
pub mod worker;

pub use server::{ServerConfig, serve};
