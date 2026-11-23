//! HTTP server for Inkling 2 Mini: OpenAI, Anthropic and Ollama APIs over a
//! single model slot, with a prefix cache in RAM and on disk.

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
