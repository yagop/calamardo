//! What the server prints when it starts: configuration and environment.

use std::ffi::{CString, c_char, c_void};

use crate::server::ServerConfig;

unsafe extern "C" {
    fn sysctlbyname(
        name: *const c_char,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *mut c_void,
        newlen: usize,
    ) -> i32;
}

/// A sysctl string value (empty if missing).
fn sysctl_string(name: &str) -> String {
    let Ok(n) = CString::new(name) else {
        return String::new();
    };
    let mut len = 0usize;
    // SAFETY: the first call asks for the size, the second fills a buffer
    // of that size.
    unsafe {
        if sysctlbyname(
            n.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        ) != 0
            || len == 0
        {
            return String::new();
        }
        let mut buf = vec![0u8; len];
        if sysctlbyname(
            n.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return String::new();
        }
        buf.truncate(len);
        String::from_utf8_lossy(&buf)
            .trim_end_matches('\0')
            .to_string()
    }
}

/// A sysctl integer value.
fn sysctl_u64(name: &str) -> Option<u64> {
    let n = CString::new(name).ok()?;
    let mut v = 0u64;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: an integer value into an 8-byte buffer; smaller ones fill its
    // low bytes (little endian).
    let rc = unsafe {
        sysctlbyname(
            n.as_ptr(),
            (&mut v as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0).then_some(v)
}

fn gb(bytes: f64) -> String {
    format!("{:.1} GB", bytes / 1e9)
}

/// Prints the configuration and the environment, before the model loads.
pub fn print_startup(cfg: &ServerConfig) {
    let p = |label: &str, value: String| eprintln!("  {label:<18} {value}");
    eprintln!(
        "Calamardo {} · MLX {}",
        env!("CARGO_PKG_VERSION"),
        calamardo_model::ops::mlx_version()
    );

    // Environment.
    let os = sysctl_string("kern.osproductversion");
    let build = sysctl_string("kern.osversion");
    let chip = sysctl_string("machdep.cpu.brand_string");
    let cores = sysctl_u64("hw.ncpu").unwrap_or(0);
    let ram = sysctl_u64("hw.memsize").unwrap_or(0) as f64;
    p(
        "system",
        format!(
            "macOS {os} ({build}) · {chip} · {cores} CPU cores · {} RAM",
            gb(ram)
        ),
    );
    let gpu = calamardo_model::ops::gpu_info();
    let get = |k: &str| gpu.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
    let mut g = get("device_name").unwrap_or_else(|| "?".into());
    if let Some(a) = get("architecture") {
        g += &format!(" ({a})");
    }
    if let Some(w) = get("max_recommended_working_set_size").and_then(|v| v.parse::<f64>().ok()) {
        g += &format!(" · recommended working set {}", gb(w));
    }
    if let Some(w) = sysctl_u64("iogpu.wired_limit_mb").filter(|&w| w > 0) {
        g += &format!(" · wired limit {}", gb(w as f64 * 1e6));
    }
    p("gpu", g);
    let mut vars: Vec<String> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("CALAMARDO_") || k.starts_with("MLX_") || k.starts_with("MTL_")
        })
        .map(|(k, v)| {
            if k.contains("KEY") {
                format!("{k}=***")
            } else {
                format!("{k}={v}")
            }
        })
        .collect();
    vars.sort();
    if !vars.is_empty() {
        p("environment", vars.join(" "));
    }

    // Configuration.
    p(
        "model",
        format!("{} ({})", cfg.model_source, cfg.model_dir.display()),
    );
    p(
        "server",
        format!(
            "http://{} · API key {} · queue {} (wait up to {} s) · default max_tokens {} · reasoning effort {}",
            cfg.addr,
            if cfg.api_key.is_some() {
                "required"
            } else {
                "not required"
            },
            cfg.queue,
            cfg.max_queue_wait.as_secs(),
            cfg.default_max_tokens,
            cfg.default_reasoning_effort
                .as_deref()
                .unwrap_or("template default"),
        ),
    );
    p(
        "kv cache",
        format!(
            "global passes {} · sliding windows bf16 · prefill chunks of {} tokens",
            if cfg.kv.is_quantized() {
                format!("{}-bit (groups of {})", cfg.kv.bits, cfg.kv.group_size)
            } else {
                "bf16".into()
            },
            cfg.prefill_step
        ),
    );
    p(
        "prefix cache",
        format!(
            "RAM {}{} · {}",
            gb(cfg.cache_budget as f64),
            if cfg.aligned_prefix {
                " (aligned reuse only)"
            } else {
                ""
            },
            match &cfg.disk_cache {
                Some((dir, budget)) => format!(
                    "disk {} up to {}, {}",
                    dir.display(),
                    gb(*budget as f64),
                    if cfg.keep_disk_cache {
                        "kept after exit"
                    } else {
                        "deleted at exit"
                    }
                ),
                None => "disk off".into(),
            }
        ),
    );
    p(
        "speculative",
        if cfg.speculative && !cfg.aligned_prefix {
            format!(
                "DSpark drafter if present (blocks up to {})",
                cfg.draft_block
            )
        } else if cfg.speculative {
            "off (aligned prefixes need reproducible output)".into()
        } else {
            "off".into()
        },
    );
}
