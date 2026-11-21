# Calamardo — requisitos del runtime para Inkling 2 Mini

Calamardo es un binario en Rust que sirve Inkling 2 Mini en un Mac Studio M4 Max de 48 GB, con caché KV de 4, 8 o 16 bits y una API compatible con OpenAI. Recoge lo aprendido al adaptar Ollama 0.40.17: contextos de más de 160k tokens sin perder la caché y sin picos de memoria.

## Objetivo y alcance

Calamardo sirve un solo modelo (Inkling 2 Mini, checkpoint `inkling-2-mini:26b-a5b-nvfp4`) en una sola máquina: Mac Studio M4 Max, GPU de 40 núcleos, 48 GB de memoria unificada, Metal 4.

**Dentro del alcance**

- Un binario en Rust sobre MLX, mediante `mlx-rs` / `mlx-sys` (MLX-C), con kernels Metal compilados en tiempo de ejecución. Sin Xcode.
- Generación de texto con razonamiento (`reasoning_effort`) y llamadas a herramientas.
- Caché KV configurable de 4, 8 o 16 bits.
- Caché de prefijos en RAM y en disco.
- Decode especulativo con el drafter DSpark del modelo.
- API compatible con OpenAI, para clientes de la red local.

**Fuera del alcance (primera versión)**

- Otros modelos, otras máquinas, CUDA o CPU.
- Visión: se carga sin la torre de imágenes, para ahorrar memoria.
- Varias conversaciones concurrentes: un solo slot.
- Entrenamiento o cuantización de pesos: se usan los pesos nvfp4 publicados.

## Arquitectura

```
 Clientes (agentes y chats de la red local)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ API HTTP compatible con OpenAI                              │
│ /v1/chat/completions · /v1/models · /metrics · sin API key  │
└─────────────────────────────────────────────────────────────┘
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Planificador                                                │
│ plantilla y tokenizador · un slot · cola con 429            │
└─────────────────────────────────────────────────────────────┘
        ▼
┌─────────────────────────────────────────────────────────────┐      ┌───────────────────────────────┐
│ Caché de prefijos                                           │ ◄──► │ Caché en disco (SSD)          │
│ trie con instantáneas · restaura desde cualquier posición   │      │ safetensors, hasta 50 GB      │
└─────────────────────────────────────────────────────────────┘      │ se expulsa aquí, no se pierde │
        ▼                                                            └───────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ Motor de Inkling 2 Mini                                     │
│ 48 pasadas · MoE nvfp4 · drafter DSpark                     │
└─────────────────────────────────────────────────────────────┘
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Caché KV configurable                                       │
│ 8 pasadas globales a 4, 8 o 16 bits · 40 ventanas de 1.024  │
│ en bf16                                                     │
└─────────────────────────────────────────────────────────────┘
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Kernels Metal en tiempo de ejecución                        │
│ prefill acotado · decode sobre caché cuantizada ·           │
│ MoE fusionado                                               │
└─────────────────────────────────────────────────────────────┘
        ▼
┌─────────────────────────────────────────────────────────────┐
│ MLX (mlx-rs / mlx-sys)                                      │
│ GPU M4 Max · 40 núcleos · 48 GB de memoria unificada        │
└─────────────────────────────────────────────────────────────┘
```

Cada petición baja de la API al planificador, que busca el prefijo en la caché (en RAM o en disco) antes de ejecutar el modelo. Solo la caché KV cambia de formato según la configuración; todo lo demás es fijo para este modelo y esta máquina.

## Requisitos del modelo

El runtime debe reproducir la arquitectura completa, con salidas iguales a una referencia (MLX en Python u Ollama) en decodificación greedy con caché de 16 bits.

| ID | Requisito | Prioridad |
| --- | --- | --- |
| M-1 | 24 capas físicas; las 12 centrales se ejecutan 3 veces: 48 pasadas por token | Obligatorio |
| M-2 | Cada pasada tiene su propia caché: 8 pasadas globales (capas 5, 11, 17 y 23 y sus repeticiones) y 40 de ventana deslizante de 1.024 tokens | Obligatorio |
| M-3 | 8 cabezas KV, `head_dim` 128 y 24 cabezas de consulta | Obligatorio |
| M-4 | Atención con bias de posición relativa aprendido y sink por consulta | Obligatorio |
| M-5 | MoE con expertos nvfp4 cargados del checkpoint sin reconvertir | Obligatorio |
| M-6 | Mezcla de residuales entre bloques y controlador del bucle (gain y forget), como en el checkpoint | Obligatorio |
| M-7 | Drafter DSpark de 3 capas (una global, con caché bf16), con longitud de borrador adaptativa | Obligatorio |
| M-8 | `reasoning_effort`: none, low, medium y high, mapeados al mensaje de sistema de esfuerzo que espera la plantilla | Obligatorio |
| M-9 | Tokenizador y plantilla de chat del checkpoint, con renderizado estable: cada prompt es prefijo exacto del siguiente | Obligatorio |
| M-10 | Llamadas a herramientas y parser de razonamiento propios de Inkling 2 | Obligatorio |
| M-11 | Visión | Fuera de la primera versión |

## Caché KV configurable

La caché de las 8 pasadas globales se elige al arrancar entre 16, 8 y 4 bits (`--kv-bits` o `CALAMARDO_KV_BITS`); por defecto, 8. Las 40 pasadas de ventana deslizante guardan solo 1.024 tokens cada una y se quedan en bf16: ocupan unos 168 MB fijos.

| Formato | Bytes por token (8 pasadas globales, K+V) | 160k tokens | 486k tokens | Calidad esperada |
| --- | --- | --- | --- | --- |
| 16 bits (bf16) | 32.768 | 5,2 GB | 15,9 GB | Referencia |
| 8 bits (afín, grupo 64) | 17.408 | 2,8 GB | 8,5 GB | Probada en Ollama: respuestas correctas |
| 4 bits (afín, grupo 64) | 9.216 | 1,5 GB | 4,5 GB | Por medir; riesgo con contextos largos |

El drafter añade 4.096 bytes por token en bf16 (0,66 GB a 160k).

**Requisitos**

1. Cuantización afín de MLX: pesos empaquetados en `uint32`, escala y bias en bf16 por grupo de 64 canales. Grupos de 32, 64 o 128 configurables.
2. Al leer la caché, cada valor se redondea al tipo de la escala, igual que `mlx.Dequantize`. Así, el kernel directo y la ruta descomprimida dan el mismo resultado.
3. Decode: un kernel que atiende directamente sobre la caché empaquetada, sin descomprimirla, para hasta 8 consultas (decode y verificación especulativa).
4. Prefill: el mismo principio, un kernel que lee 4 u 8 bits sin descomprimir el historial entero en cada pasada global. En Ollama esa descompresión era una de las causas del pico de memoria.
5. Una instantánea solo se restaura en una caché con los mismos bits y tamaño de grupo. Con otro formato se rechaza.
6. Los buffers de la caché se reservan por bloques y se leen como vistas: una instantánea nunca comparte arrays vivos que se sigan escribiendo.

## Caché de prefijos

En cada turno, una conversación debe procesar solo los tokens nuevos, aunque la caché tenga que salir de la RAM. En Ollama, un fallo de caché con 164k tokens suponía unos 15 minutos de prefill; leer esa misma caché desde el SSD cuesta alrededor de 1 segundo.

| ID | Requisito | Prioridad |
| --- | --- | --- |
| C-1 | Árbol (trie) de prefijos con instantáneas cada 8.192 tokens y al final de cada prompt | Obligatorio |
| C-2 | Restaurar desde cualquier instantánea, sin alinear a bloques de 2.048 tokens. La alineación estricta (necesaria para salidas bit a bit, por el enrutado del MoE) queda como opción | Obligatorio |
| C-3 | Nivel en disco: cuando la memoria obliga a expulsar una conversación, se escribe en el SSD en vez de perderse. Al restaurar, se lee de disco si no está en RAM | Obligatorio |
| C-4 | Formato en disco: safetensors con la ruta de tokens como clave, más un manifiesto con modelo, bits, tamaño de grupo y versión. Si algo no coincide, se descarta | Obligatorio |
| C-5 | Presupuesto de disco configurable (por defecto 50 GB), expulsando lo usado hace más tiempo | Obligatorio |
| C-6 | La política de memoria reserva el pico real del prefill y la caché viva, y nunca vacía la caché por la presión de compresión de macOS sin una opción explícita | Obligatorio |
| C-7 | Las peticiones canceladas por el cliente conservan lo procesado: el siguiente reintento continúa desde la última instantánea | Obligatorio |
| C-8 | El modelo no se descarga por inactividad salvo que se configure (equivalente a `keep_alive` infinito) | Obligatorio |
| C-9 | Precalentar al arrancar: cargar desde disco la última conversación usada | Deseable |

## Kernels y memoria

La memoria transitoria del prefill no debe crecer con el contexto. En Ollama, a 72k tokens el prefill añadía 19,3 GiB porque MLX creaba a la vez las tablas de bias de todos los bloques de consultas. Ordenando los bloques y acotándolos, el pico bajó a 2,6 GiB con salidas idénticas.

| ID | Requisito | Prioridad |
| --- | --- | --- |
| K-1 | Prefill de atención relativa por bloques de consultas: cada bloque espera a que el anterior termine (`mlx_depends`), y el tamaño del bloque (256, 128 o 64 filas) se elige para que una tabla de bias no pase de 512 MiB | Obligatorio |
| K-2 | Objetivo: un kernel de prefill tipo flash que calcule el bias sobre la marcha y lea la caché cuantizada, sin tabla en memoria ni historial descomprimido | Obligatorio |
| K-3 | Las pasadas globales de un mismo bloque de prefill no pueden tener su historial descomprimido a la vez | Obligatorio |
| K-4 | Decode: kernel de atención dividido en 64 partes en M4 Max (`applegpu_g16s`), igual que en M4 Pro | Obligatorio |
| K-5 | Kernels Metal compilados en tiempo de ejecución (`fast::metal_kernel`) con `ensure_row_contiguous`, sin Xcode ni `metallib` precompilado | Obligatorio |
| K-6 | Fusionar las operaciones pequeñas de cada pasada (mezcla de residuales, controlador del bucle, normalizaciones) y compilar cada pasada con `compile`, para reducir lanzamientos de kernels | Deseable |
| K-7 | Kernel MoE fusionado para un token con expertos nvfp4 (gate+up en una pasada) | Deseable |
| K-8 | Cada kernel nuevo se compara contra la ruta de referencia de MLX y se exige igualdad exacta, o una tolerancia documentada si cambia el redondeo | Obligatorio |
| K-9 | Medir el pico de cada paso (`mlx_get_peak_memory`) y exponerlo en los logs y en `/metrics` | Obligatorio |

## API compatible

Cualquier cliente que hoy habla con Ollama por `/v1` debe funcionar sin cambios, salvo la URL.

| Endpoint | Compatible con | Prioridad |
| --- | --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions: streaming SSE, `tools`, `tool_choice`, `reasoning_effort`, `max_tokens`, `temperature`, `top_p`, `seed`, `stop` | Obligatorio |
| `GET /v1/models` | OpenAI | Obligatorio |
| `POST /v1/completions` | OpenAI (texto plano) | Deseable |
| `POST /v1/responses` | OpenAI Responses | Deseable |
| `POST /v1/messages` | Anthropic Messages, para clientes tipo Claude Code | Deseable |
| `POST /api/chat`, `GET /api/tags`, `GET /api/ps` | Ollama, solo lo que usan los clientes actuales | Deseable |
| `GET /health`, `GET /metrics` | Estado y métricas Prometheus | Obligatorio |

**Requisitos**

1. `usage` incluye los tokens del prompt, los generados y los servidos desde caché (`prompt_tokens_details.cached_tokens`).
2. Escucha en una dirección configurable (por defecto `127.0.0.1:11434`; se puede poner `0.0.0.0` para la red interna).
3. Sin API key por defecto. La autenticación por token es opcional y se activa con una variable.
4. Sin conexiones salientes: no hay telemetría, recomendaciones de modelos ni búsqueda web.
5. Si el cliente cierra la conexión, la generación se detiene pero se conserva el prefill hecho (C-7).
6. El progreso del prefill se envía en streaming, para que el cliente no corte por inactividad durante prefills largos.
7. Un solo slot. Las peticiones concurrentes esperan en cola, y cuando la espera es excesiva se devuelve 429.

## Objetivos de rendimiento

Calamardo debe igualar a nuestro Ollama modificado con contexto corto y superarlo claramente con contextos largos, que es donde falla hoy. La línea base se midió en esta misma máquina el 21 de noviembre de 2026, con Ollama 0.40.17 parcheado y caché de 8 bits salvo donde se indica.

| Métrica | Ollama medido | Objetivo Calamardo |
| --- | --- | --- |
| Prefill, prompt de 29k (caché de 16 bits) | 570 tok/s | ≥ 600 tok/s |
| Prefill a ~145k de contexto | ~90 tok/s | ≥ 200 tok/s |
| Prefill en frío de 164k tokens | ~15 min | ≤ 7 min |
| Decode con 29k de contexto y drafter (caché de 16 bits) | 36,7 tok/s (34–41) | ≥ 40 tok/s |
| Turno de ~350 tokens nuevos sobre 36k ya en caché | 1,5 s | ≤ 1,5 s |
| Pico transitorio por encima de lo activo a ~160k | 9–10 GB | ≤ 3 GB |
| Contexto que se mantiene en caché entre turnos | ~150k tokens | ≥ 300k (RAM + disco) |
| Restaurar 160k tokens desde disco | No existe | ≤ 2 s |
| Aceptación del drafter | 52–83 % | Igual o mejor |

## Verificación

Nada entra sin un test que compare contra una referencia y sin un benchmark antes y después.

1. **Exactitud de kernels:** cada kernel se compara con la ruta de MLX en bf16 y fp32, con formas pequeñas y grandes, vistas con desplazamiento, GQA y ventana. Igualdad exacta, o tolerancia documentada si cambia el redondeo.
2. **Exactitud del modelo:** decodificación greedy de 256 tokens igual que la referencia con caché de 16 bits. Con 8 y 4 bits, medir la divergencia KL contra 16 bits y fijar un umbral.
3. **Prueba de aguja:** un documento de 29k tokens con un código escondido (el "4817" de las pruebas en Ollama), y versiones de 100k y 160k. Las tres configuraciones de caché deben encontrarlo.
4. **Caché:** una conversación que crece de 30k a 200k tokens en turnos pequeños y grandes. Se comprueba que cada turno procesa solo lo nuevo y que una expulsión a disco se recupera sin recalcular.
5. **Memoria:** el pico transitorio por paso queda registrado y no supera el objetivo a 160k.
6. **Benchmark reproducible:** un comando que mide prefill y decode con prompts únicos (sin caché) y alterna versiones, como hicimos con `main` y la rama del M4 Max.
7. **Cliente real:** el cliente de la red local (192.168.0.171) usado durante una semana sin errores de tiempo de espera.

## Hitos

Cada hito termina con sus tests y benchmarks en verde, y no empieza el siguiente hasta entonces.

1. **Base:** carga de pesos nvfp4 con `mlx-rs`, las 48 pasadas y una salida greedy igual a la referencia con caché de 16 bits.
2. **Caché KV de 8 y 4 bits** con kernel de decode directo (K-4, requisitos de la caché KV).
3. **Prefill acotado:** bloques ordenados y de tamaño adaptativo (K-1, K-3).
4. **Servidor:** `/v1/chat/completions` con streaming, herramientas y razonamiento; métricas.
5. **Caché de prefijos en RAM**, con restauración sin alinear (C-1, C-2, C-7, C-8).
6. **Caché en disco** (C-3 a C-5).
7. **Drafter DSpark** con longitud adaptativa.
8. **Kernel de prefill tipo flash** sobre la caché cuantizada (K-2), y fusiones (K-6, K-7).

## Preguntas abiertas

- ¿La calidad con 4 bits es aceptable a 160k tokens? Hay que medirla antes de ofrecerla como opción recomendada.
- ¿Madurez de `mlx-rs`? Está en desarrollo activo: confirmar que expone todo lo necesario (`metal_kernel`, cuantización, `compile`, `depends`) o si hace falta llamar a `mlx-sys` directamente.
- ¿Rendimiento de lectura y escritura del SSD con instantáneas de 3–4 GB, y su desgaste con el uso diario?
- ¿Mantener compatibilidad con el formato de pesos de Ollama (manifiestos y blobs) o leer directamente el checkpoint de Hugging Face?
- ¿Dejar abierta la puerta a un segundo slot si algún día hay dos clientes?

## Notas de revisión (21 de noviembre de 2026)

Comprobado contra el checkpoint oficial y `mlx-rs` 0.34.0 / `mlx-sys` 0.7.0 (MLX v0.32.2). Pendiente de decidir: nada de esto cambia los requisitos de arriba hasta que se acepte.

**Checkpoint**

- Se usa el repositorio oficial `thinkingmachines/Inkling-2-Mini` de Hugging Face, leído desde la caché estándar del Hub. Nada de Ollama. Resuelve la pregunta abierta sobre el formato de pesos.
- Un shard de 19,53 GB (`model-00001-of-00001.safetensors`) y el drafter aparte en `mtp.safetensors` (0,64 GB). El modelo principal sin visión ocupa unos 18,9 GB.
- M-1/M-2: `loop_config` fija `core_start = 6`, `core_end = 18`, `num_passes = 3`. Las capas 6–17 se repiten; las globales son las que no están en `local_layer_ids`: 5, 11, 17 y 23. Pasadas globales: 5, 11, 17, 23, 29, 35, 41 y 47.
- El drafter lee los estados de las pasadas 5, 23, 41 y 48 (`mtp_config.target_layer_ids`); sus capas locales son la 0 y la 2, la 1 es global.

**M-5 hay que corregirlo: los expertos no son todos nvfp4**

- Capas 0–19: nvfp4. Pesos E2M1 empaquetados en `u8` (`[128, 2304, 1152]` para `w13`), escala E4M3 por bloque de 16 y escala externa por experto (`*_amax` / `*_outer_scale_denominator`).
- Capas 20–23: mxfp8. Pesos E4M3, escala UE8M0 por bloque de 32 (`weight_block_shape` 32×32).
- Los bytes valen tal cual para MLX: el `u8` empaquetado se reinterpreta como `u32` sin reconvertir.
- `hf_quant_config.json` declara `required_activation_quantization` en los expertos: activaciones nvfp4 por token (bloque 16, regla `fouroversix_mae`) en las capas 0–19 y mxfp8 en las 20–23. El modelo se entrenó así (W4A4).
- **Decidido: W4A16.** Las activaciones de los expertos no se cuantizan; entran en bf16 a `gather_qmm`, como en el port de MLX y en Ollama. El M4 Max no tiene aritmética fp4/fp8, así que cuantizarlas no acelera nada y añade trabajo. La referencia para la exactitud del modelo (verificación 2) es una implementación MLX W4A16; contra vLLM o SGLang (W4A4) se esperan pequeñas diferencias. Si se sospecha de la calidad, se puede añadir un modo que simule A4 y medirlo.
- Las consultas de la mezcla de residuales (`attn_res.*_queries`) se calculan en f32, y los logits de `unembed` salen en f32.

**MLX y la compilación**

- `mlx-rs` incluye `fast::MetalKernel` (con `ensure_row_contiguous`), `compile`, `quantize`/`dequantize`/`quantized_matmul`/`gather_qmm` con nvfp4 y mxfp8, `memory::peak_memory` y `set_wired_limit`. Falta `depends`, que sí está en MLX-C (`mlx_depends`).
- `mlx-sys` compila MLX desde el código fuente con CMake. Incluso con `MLX_METAL_JIT`, MLX 0.32.2 precompila 9 kernels con `xcrun metal` (arg_reduce, conv, dot, layer_norm, random, rms_norm, rope, sdpa, fence). Sin Xcode, eso no compila.
- Decidido: no se compila MLX. `scripts/fetch-mlx.sh` descarga la rueda oficial `mlx-metal==0.32.4` de PyPI (sha256 fijado) y deja `libmlx.dylib`, `mlx.metallib` y las cabeceras en `third_party/mlx`. Hace falta 0.32.4 (con MLX-C 0.8.0) porque `gather_qmm` recibe ahí la escala global de nvfp4; MLX-C 0.8.0 cambió esa firma y `crates/mlx-sys/vendor/compat` mantiene la de 0.7.0 para `mlx-rs` 0.34. `crates/mlx-sys` sustituye al `mlx-sys` de crates.io (`[patch.crates-io]`): compila solo MLX-C con `cc` y genera los bindings con bindgen, así que `mlx-rs` funciona encima sin cambios. Sin CMake, sin compilador Metal y sin Xcode.
- Verificado en la GPU (`crates/calamardo/tests/mlx_smoke.rs`): kernel Metal compilado en tiempo de ejecución, `quantized_matmul` nvfp4 y mxfp8 frente a la ruta descomprimida, `mlx_depends` por FFI y `peak_memory`.
- K-5 se refiere a los kernels propios; el metallib de MLX viene precompilado con la librería.

**Hito 2: caché KV de 8 y 4 bits (estado)**

- `--kv-bits` / `CALAMARDO_KV_BITS` (16, 8 o 4; por defecto 8) y `--kv-group` / `CALAMARDO_KV_GROUP` (32, 64 o 128; por defecto 64). Solo las 8 pasadas globales; las ventanas siguen en bf16. Con 16 bits el camino es el del Hito 1 y la paridad exacta con la referencia se mantiene.
- Requisito 1: cuantización afín de MLX (`mx.quantize`), u32 empaquetado y escala y bias bf16 por grupo. Cumplido.
- Requisito 2: el kernel descomprime con el mismo resultado que `mx.dequantize`, bit a bit (test `dequantization_matches_mlx_bit_for_bit`, 4 y 8 bits, grupos 32/64/128). En Metal, `scale * q + bias` en bfloat se evalúa en float y redondea una vez; el kernel lo hace explícitamente, un 30 % más rápido que la aritmética bfloat.
- Requisito 3: decode de hasta 8 consultas directamente sobre la caché empaquetada (`quant_decode_partial` + `quant_decode_reduce`). K-8: frente a una referencia en float64 sobre la caché descomprimida, error ≤ 2 ulps de bf16 + 1e-3 (el peor caso usa el 39 % de ese margen).
- K-4: el número de partes se adapta al contexto (≥ 256 claves por parte, máximo 512). Con este kernel, 64 partes son 1,5 veces más lentas en M4 Max: a 160k tokens, 2,08 ms por pasada con 64 partes frente a 1,38 ms con 512 (8 bits), 1,26–1,38 ms (4 bits) y 1,88 ms del `sdpa` de MLX sobre bf16.
- Requisito 4 (prefill sin descomprimir): pendiente. El prefill descomprime la caché de cada pasada global y usa el kernel relativo; es K-2 (Hito 8). K-3 (no tener a la vez el historial descomprimido de varias pasadas) es del Hito 3.
- Requisitos 5 y 6: la caché guarda su formato (bits y grupo) y crece por bloques de 256 tokens; la comprobación al restaurar y las vistas de las instantáneas llegan con la caché de prefijos (Hitos 5 y 6).
- Calidad (verificación 2), 8.042 tokens de prompt y 256 generados forzando los tokens de bf16 (`calamardo kv-eval`): KL media 0,0044 (8 bits) y 0,0076 (4 bits/64, igual con grupo 32); el top-1 coincide en 251 y 248 de 256 tokens. La aguja aparece con los tres formatos a 8k y 31k. Falta medir a 100k y 160k, después del Hito 3 (hoy el pico de memoria del prefill lo impide).
- Rendimiento con 31k de contexto: prefill a 784 tok/s y decode a 32 tok/s sin drafter en los tres formatos.

**Hito 3: prefill acotado (estado)**

- Causa del pico: el forward de un trozo de 2.048 tokens se construía como un solo grafo de 48 pasadas y MLX mantenía vivos los intermedios de muchas a la vez: unos +4 GB por trozo, constantes con el contexto. En esta implementación la tabla de bias es L × 24 × 1.024 (unos 100 MB por pasada con L = 2.048) y no crece con el contexto, así que el problema de Ollama que describe K-1 no aparece aquí.
- K-1 y K-3: con más de 8 tokens, el forward evalúa al final de cada pasada su salida, sus estados de bloque y su caché, antes de construir la siguiente (`Model::bounded_prefill_from`). Solo están vivos los intermedios de una pasada, y como mucho un historial descomprimido. Es una barrera de evaluación en vez de `mlx_depends`; mismo efecto y los valores no cambian (paridad con la referencia intacta). El pico de un trozo de 2.048 baja de +4,0 a +1,6 GB sin perder velocidad; con trozos de 4.096, +2,2 GB y un 5 % más rápido.
- K-9 (parcial): cada trozo de prefill informa de tokens, tiempo, memoria activa y pico (`calamardo generate -v`). Falta exponerlo en `/metrics` (Hito 4).
- Medido con un prompt de 166.561 tokens y caché de 8 bits:
  - Prefill en frío: 5,3 min (objetivo ≤ 7; Ollama ~15).
  - Prefill a ~145k de contexto: 401 tok/s (objetivo ≥ 200; Ollama ~90). Va de 816 tok/s al principio a 378 al final.
  - Pico transitorio por encima de lo activo: 2,13 GB como máximo (objetivo ≤ 3; Ollama 9–10). Pico total, 24,4 GB.
  - Decode a 166k: 26,6 tok/s sin drafter. Encuentra la aguja.
- Calidad a 166k (`kv-eval`, 128 tokens forzados desde bf16): KL media 0,0026 con 8 bits (126/128 en top-1) y 0,0102 con 4 bits (125/128). A 8k eran 0,0044 y 0,0076: 8 bits no empeora con el contexto y 4 bits sí, alrededor de 4 veces la divergencia de 8 bits a 160k. Responde la pregunta abierta: 4 bits funciona, pero la recomendación sigue siendo 8 bits y 4 bits queda como opción para contextos que no caben de otra forma.

**Incoherencias a resolver**

1. Con un coste por token lineal en el contexto (solo crecen las 8 pasadas globales), 200 tok/s a 145k implican unos 9,5 min para 164k en frío. Para ≤ 7 min hacen falta unos 300 tok/s a 145k. Resuelto en la práctica: Calamardo hace 401 tok/s a 145k y 5,3 min en frío.
2. C-1 con instantáneas completas cada 8.192 tokens acumula unos 30–40 GB por conversación de 160k. La caché global solo crece por el final: los bloques sellados deben compartirse entre instantáneas, y solo copiarse las ventanas (unos 168 MB). Matiza el requisito 6 de la caché KV.
3. El puerto por defecto 11434 choca con Ollama durante las comparativas (verificación 6).
4. Restaurar 160k desde disco: unos 3,6 GB a 8 bits con drafter y ventanas. Cabe en 2 s solo con lectura directa (mmap) a los buffers.
