//! Streams Inkling 2 output into reasoning, content and tool calls (M-10),
//! following the checkpoint's `response_template`:
//!
//! ```text
//! <|message_model|><|content_thinking|>…<|end_message|>
//! <|message_model|><|content_text|>…<|end_message|>
//! <|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>
//! <|content_model_end_sampling|>
//! ```

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

/// Ids of the special tokens the parser reacts to.
#[derive(Clone, Copy, Debug)]
pub struct Specials {
    pub message_model: u32,
    pub content_thinking: u32,
    pub content_text: u32,
    pub content_tool_error: u32,
    pub content_invoke_tool_text: u32,
    pub content_invoke_tool_json: u32,
    pub end_message: u32,
    pub end_sampling: u32,
}

impl Specials {
    pub fn from_tokenizer(t: &Tokenizer) -> Result<Self> {
        let id = |s: &str| {
            t.token_to_id(s)
                .with_context(|| format!("tokenizer has no {s}"))
        };
        Ok(Self {
            message_model: id("<|message_model|>")?,
            content_thinking: id("<|content_thinking|>")?,
            content_text: id("<|content_text|>")?,
            content_tool_error: id("<|content_tool_error|>")?,
            content_invoke_tool_text: id("<|content_invoke_tool_text|>")?,
            content_invoke_tool_json: id("<|content_invoke_tool_json|>")?,
            end_message: id("<|end_message|>")?,
            end_sampling: id("<|content_model_end_sampling|>")?,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    /// A complete tool call; `arguments` is a JSON object as a string.
    ToolCall {
        name: String,
        arguments: String,
    },
}

/// Turns token ids into text without splitting multi-token characters.
#[derive(Default)]
pub struct TextDecoder {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl TextDecoder {
    pub fn push(&mut self, t: &Tokenizer, id: u32) -> Option<String> {
        self.ids.push(id);
        let printed = t.decode(&self.ids[self.prefix..self.read], false).ok()?;
        let current = t.decode(&self.ids[self.prefix..], false).ok()?;
        if current.len() > printed.len() && !current.ends_with('\u{FFFD}') {
            let new = current[printed.len()..].to_string();
            self.prefix = self.read;
            self.read = self.ids.len();
            Some(new)
        } else {
            None
        }
    }

    /// Whatever is left, even if incomplete.
    pub fn finish(&mut self, t: &Tokenizer) -> Option<String> {
        let printed = t.decode(&self.ids[self.prefix..self.read], false).ok()?;
        let current = t.decode(&self.ids[self.prefix..], false).ok()?;
        *self = Self::default();
        (current.len() > printed.len()).then(|| current[printed.len()..].to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    /// Between messages.
    Idle,
    /// After `<|message_model|>`, reading the optional name.
    Header,
    Reasoning,
    Content,
    ToolJson,
}

pub struct ResponseParser {
    sp: Specials,
    state: State,
    text: TextDecoder,
    header: Vec<u32>,
    tool: Vec<u32>,
    tool_name: String,
    pub finished: bool,
}

impl ResponseParser {
    pub fn new(sp: Specials) -> Self {
        Self {
            sp,
            state: State::Idle,
            text: TextDecoder::default(),
            header: Vec::new(),
            tool: Vec::new(),
            tool_name: String::new(),
            finished: false,
        }
    }

    fn text_delta(&self, s: String) -> Delta {
        match self.state {
            State::Reasoning => Delta::Reasoning(s),
            _ => Delta::Content(s),
        }
    }

    fn close(&mut self, t: &Tokenizer, out: &mut Vec<Delta>) {
        match self.state {
            State::Reasoning | State::Content => {
                if let Some(s) = self.text.finish(t) {
                    out.push(self.text_delta(s));
                }
            }
            State::ToolJson => {
                let raw = t.decode(&self.tool, false).unwrap_or_default();
                self.tool.clear();
                out.push(parse_tool_call(&raw, &self.tool_name));
            }
            State::Header | State::Idle => {}
        }
        self.state = State::Idle;
    }

    pub fn push(&mut self, t: &Tokenizer, id: u32) -> Vec<Delta> {
        let mut out = Vec::new();
        let sp = self.sp;
        if id == sp.end_sampling {
            self.close(t, &mut out);
            self.finished = true;
        } else if id == sp.end_message {
            self.close(t, &mut out);
        } else if id == sp.message_model {
            self.close(t, &mut out);
            self.state = State::Header;
            self.header.clear();
        } else if id == sp.content_thinking {
            self.close(t, &mut out);
            self.state = State::Reasoning;
        } else if id == sp.content_text
            || id == sp.content_tool_error
            || id == sp.content_invoke_tool_text
        {
            self.close(t, &mut out);
            self.state = State::Content;
        } else if id == sp.content_invoke_tool_json {
            self.tool_name = t
                .decode(&self.header, false)
                .unwrap_or_default()
                .trim()
                .to_string();
            self.close(t, &mut out);
            self.state = State::ToolJson;
        } else {
            match self.state {
                State::Header => self.header.push(id),
                State::ToolJson => self.tool.push(id),
                State::Idle => {
                    // Text outside a message: treat it as content.
                    self.state = State::Content;
                    if let Some(s) = self.text.push(t, id) {
                        out.push(Delta::Content(s));
                    }
                }
                State::Reasoning | State::Content => {
                    if let Some(s) = self.text.push(t, id) {
                        out.push(self.text_delta(s));
                    }
                }
            }
        }
        out
    }

    /// Flushes an unterminated message (generation cut by max_tokens).
    pub fn finish(&mut self, t: &Tokenizer) -> Vec<Delta> {
        let mut out = Vec::new();
        if self.state != State::ToolJson {
            self.close(t, &mut out);
        }
        out
    }
}

/// `{"name": ..., "args": {...}}` into a tool call; falls back to the header
/// name and the raw text if the JSON does not parse.
fn parse_tool_call(raw: &str, header_name: &str) -> Delta {
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(v) => {
            let name = v
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or(header_name)
                .to_string();
            let args = v.get("args").cloned().unwrap_or(serde_json::json!({}));
            Delta::ToolCall {
                name,
                arguments: args.to_string(),
            }
        }
        Err(_) => Delta::ToolCall {
            name: header_name.to_string(),
            arguments: raw.to_string(),
        },
    }
}
