//! OpenAI-compatible server for Inkling 2 Mini (milestone 4).

mod anthropic;
pub mod disk;
pub mod metrics;
mod ollama;
mod openai;
pub mod parser;
mod report;
pub mod server;
pub mod template;
pub mod worker;

pub use server::{ServerConfig, serve};
