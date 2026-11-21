//! The checkpoint's chat template (`chat_template.jinja`), rendered as
//! Hugging Face transformers renders it (M-8, M-9).

use std::path::Path;

use anyhow::{Context, Result};
use minijinja::{Environment, Error, ErrorKind, Value, value::Kwargs};
use serde_json::Value as Json;

pub struct ChatTemplate {
    env: Environment<'static>,
}

/// transformers' `tojson`: `json.dumps(x, ensure_ascii=False, indent=None,
/// separators=..., sort_keys=...)`.
fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let sort_keys: bool = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    let _: Option<bool> = kwargs.get("ensure_ascii")?;
    kwargs.assert_all_used()?;
    let (item_sep, key_sep) = match separators.as_deref() {
        Some([a, b]) => (a.clone(), b.clone()),
        Some(_) => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "separators needs two strings",
            ));
        }
        None => (", ".to_string(), ": ".to_string()),
    };
    let json = serde_json::to_value(&value)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let mut out = String::new();
    dump(&json, sort_keys, &item_sep, &key_sep, &mut out);
    Ok(Value::from_safe_string(out))
}

/// Python `json.dumps` with explicit separators.
pub fn dump(v: &Json, sort_keys: bool, item_sep: &str, key_sep: &str, out: &mut String) {
    match v {
        Json::Object(map) => {
            out.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            if sort_keys {
                keys.sort();
            }
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                out.push_str(&Json::String(k.clone()).to_string());
                out.push_str(key_sep);
                dump(&map[k], sort_keys, item_sep, key_sep, out);
            }
            out.push('}');
        }
        Json::Array(items) => {
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                dump(x, sort_keys, item_sep, key_sep, out);
            }
            out.push(']');
        }
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Null => out.push_str("null"),
        other => out.push_str(&other.to_string()),
    }
}

fn raise_exception(msg: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

impl ChatTemplate {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_source(&text)
    }

    pub fn from_source(text: &str) -> Result<Self> {
        // `{% generation %}` only marks assistant tokens for training masks
        // in transformers; it renders its body unchanged.
        let mut source = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(start) = rest.find("{%") {
            let end = rest[start..]
                .find("%}")
                .map(|e| start + e + 2)
                .context("unterminated tag")?;
            let tag =
                rest[start + 2..end - 2].trim_matches(|c: char| c == '-' || c.is_whitespace());
            source.push_str(&rest[..start]);
            if tag != "generation" && tag != "endgeneration" {
                source.push_str(&rest[start..end]);
            }
            rest = &rest[end..];
        }
        source.push_str(rest);

        let mut env = Environment::new();
        // transformers renders with trim_blocks and lstrip_blocks.
        env.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()?,
        );
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", raise_exception);
        env.add_template_owned("chat", source)?;
        Ok(Self { env })
    }

    /// Renders `messages` (OpenAI format, with tool call arguments already
    /// parsed to objects) for generation.
    pub fn render(
        &self,
        messages: &Json,
        tools: Option<&Json>,
        reasoning_effort: Option<&Json>,
    ) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        let ctx = minijinja::context! {
            messages => Value::from(minijinja::value::Serde(messages)),
            tools => tools.map(|t| Value::from(minijinja::value::Serde(t))),
            reasoning_effort => reasoning_effort.map(|r| Value::from(minijinja::value::Serde(r))),
            add_generation_prompt => true,
        };
        tmpl.render(ctx)
            .map_err(|e| anyhow::anyhow!("chat template: {e:#}"))
    }
}
