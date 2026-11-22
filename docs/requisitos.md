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
| K-7 | Kernel de expertos para pocos tokens (verificación especulativa) que lea cada experto una vez (originalmente: MoE fusionado para un token con expertos nvfp4) | Deseable |
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

**Hito 4: servidor (estado)**

- `calamardo serve` (axum). Por defecto escucha en `127.0.0.1:11434` (`--addr` / `CALAMARDO_ADDR`; `0.0.0.0:…` para la red local), con caché de 8 bits, sin API key (`CALAMARDO_API_KEY` la activa como `Authorization: Bearer`) y sin conexiones salientes.
- Endpoints: `POST /v1/chat/completions` (con y sin streaming), `GET /v1/models`, `GET /health` y `GET /metrics` (Prometheus). Se acepta cualquier nombre de modelo en las peticiones, así que los clientes que piden `inkling-2-mini:26b-a5b-nvfp4` funcionan sin cambios.
- M-8 y M-9: la plantilla oficial se renderiza con minijinja y da exactamente el mismo texto que `apply_chat_template` de transformers (test con 5 conversaciones: sistema, historial con razonamiento, herramientas con llamada y resultado, contenido por partes, esfuerzo numérico). `reasoning_effort` (o `reasoning.effort`) pasa a la plantilla; sin él, la plantilla usa 0.9 (`--default-reasoning-effort` lo cambia).
- M-10: el parser sigue el `response_template` del checkpoint, por ids de tokens especiales. El razonamiento sale en `reasoning`, el campo de Ollama; en el historial se acepta `reasoning` o `reasoning_content`. Las llamadas a herramientas salen en `tool_calls`, con `finish_reason: "tool_calls"`, y los argumentos como cadena JSON; los que llegan como cadena en el historial se convierten en objeto para la plantilla.
- Muestreo: `temperature` (0 = greedy en GPU, el mismo camino que la verificación), `top_p`, `seed`, `max_tokens`/`max_completion_tokens` (por defecto 32.768) y `stop`. Las secuencias de parada se aplican al contenido; si una llega partida entre tokens, la parte anterior puede haberse enviado ya.
- `tool_choice`: solo se respeta `"none"`; `"required"` o una función concreta no se fuerzan (haría falta decodificación restringida).
- Requisito 1: `usage` incluye `prompt_tokens_details.cached_tokens`, a 0 hasta la caché de prefijos (Hito 5).
- Requisitos 5 y 6: si el cliente corta, la generación se para en el siguiente token (probado: 89 de 2.000 tokens tras cortar a los 3 s). Conservar el prefill hecho (C-7) es del Hito 5. Durante el prefill, el stream manda un comentario SSE por trozo (`: prefill 2048/166561`), y cada 15 s otro de keep-alive.
- Requisito 7: un solo slot en un hilo dueño del modelo. Las peticiones esperan en una cola de `--queue` (8); con la cola llena devuelven 429 al instante, y tras `--max-queue-wait` (900 s) de espera, también 429 (probado).
- K-9: `/metrics` expone el pico de memoria de cada paso de prefill (máximo histórico y de la última petición), el transitorio por encima de lo activo, la memoria activa, velocidades, tokens, cola y rechazos.
- Revisión (subagente): corregidos los cortes de `stop` en mitad de un carácter UTF-8 (tumbaban la petición), `busy` atascado tras una cancelación, conversaciones bloqueadas por una llamada a herramienta con JSON inválido en el historial, pánicos del hilo del modelo (ahora se capturan y `/health` devuelve 503 si el hilo muere) y la caché de buffers de MLX sin liberar en el decode. Compatibilidad: cuerpos de hasta 256 MB, cualquier `Content-Type`, errores de JSON con la forma de OpenAI, `developer` como `system`, `GET /v1/models/{id}`, CORS para orígenes locales (como Ollama), `seed` y `max_tokens` decimales, 400 con `logprobs`, chunk de error si un stream acaba sin terminar, parada también ante `<|message_user|>`, `<|message_system|>`, `<|message_tool|>` y `<|endoftext|>`, y una cola propia que descarta al momento las peticiones abandonadas. `response_format` (`json_object`, `json_schema`) se traduce en una instrucción de sistema, sin restringir la salida. El puerto se reserva antes de cargar el modelo y la comparación del token es en tiempo constante.
- Pendiente: los endpoints deseables (`/v1/completions`, `/v1/responses`, `/v1/messages`, `/api/*` de Ollama) y la verificación 7 (una semana con el cliente de 192.168.0.171).

**Hito 5: caché de prefijos en RAM (estado)**

- Instantáneas (C-1): cada 8.192 tokens durante el prefill y al final de cada prompt, justo antes de su último token, de modo que reintentar el mismo prompt solo recalcula ese token con las mismas operaciones. Como la caché global solo crece por el final, la instantánea guarda su longitud, no una copia; las ventanas y los estados de convolución se guardan por referencia (MLX solo copia un array si se escribe mientras está compartido). Así se cumple el requisito 6 de la caché KV sin copiar la caché global por instantánea.
- Reutilización (C-2): para cada petición se busca, entre las sesiones guardadas, el punto más largo del prompt que ya tienen: el final vivo de la conversación o la instantánea más larga dentro del prefijo común, sin alinear. `--aligned-prefix` limita la reutilización a múltiplos del trozo de prefill, para que la salida sea bit a bit la de una ejecución en frío. Volver a una instantánea anterior trunca la caché; si el presupuesto lo permite, se bifurca y la conversación original se conserva (las cachés se comparten hasta que una de las dos escribe).
- Memoria (C-6): `--cache-gb` (16 por defecto) limita las sesiones inactivas y sus instantáneas; se expulsa la sesión menos usada y después las instantáneas periódicas más antiguas. No se reacciona a la presión de memoria de macOS.
- C-7: si una petición se cancela, la sesión se guarda con lo ya procesado y el reintento sigue desde ahí. C-8: el modelo no se descarga nunca.
- `usage.prompt_tokens_details.cached_tokens` informa de lo reutilizado; `/metrics` añade `cached_tokens_total`, `prefix_cache_sessions` y `prefix_cache_bytes`.
- Test con el modelo real: volver a la instantánea de fin de prompt y bifurcar desde ella reproducen exactamente los tokens greedy de una ejecución en frío, en bf16 y en 8 bits. En el servidor, con un prompt de 8k: el segundo turno reutiliza 8.224 de 8.235 tokens; un reintento idéntico, 8.234; el mismo turno con el razonamiento quitado del historial, 8.053 (instantánea del final del primer turno); volver a la primera rama, 8.434 (la bifurcación conservó la original).
- C-9 y el nivel en disco, en el Hito 6.

**Hito 6: caché en disco (estado)**

- C-3: cuando el presupuesto de RAM obliga a expulsar una conversación, se guarda en el SSD. Una petición que la continúa la recupera si el disco tiene más prefijo que la RAM; al cargarse, el archivo se borra (vuelve a disco cuando se expulse otra vez o al cerrar).
- C-4: un safetensors por conversación, con nombre por hash de su ruta de tokens. Contiene los buffers de las cachés tal cual (sin copias para trocearlos), los tokens, la última instantánea de fin de prompt y un manifiesto en los metadatos: versión de formato, modelo (hash del snapshot), bits y grupo de la caché. Al arrancar se reconstruye el índice leyendo solo cabeceras y tokens; lo que no coincide con el manifiesto se borra. Se escribe a un temporal y se renombra, así un corte nunca deja un archivo a medias.
- C-5: `--disk-cache-gb` (50 por defecto; 0 lo desactiva) y `--disk-cache-dir` (`~/.cache/calamardo/sessions`). Se expulsa primero lo usado hace más tiempo (fecha de modificación del archivo).
- C-9: al cerrar con Ctrl-C se guardan en disco las conversaciones de RAM, y al arrancar se precarga la más reciente.
- Tests con el modelo real: una sesión guardada y cargada sigue generando exactamente los mismos tokens que la original, y su instantánea de fin de prompt reproduce la ejecución en frío (bf16 y 4 bits); un manifiesto distinto se rechaza.
- Medido en el servidor con 0,3 GB de RAM para la caché (cada conversación nueva expulsa la anterior):
  - 8k tokens: guardar 0,12 s, restaurar 0,02 s; el turno siguiente reutiliza 8.093 de 8.102 tokens y responde en 1,6 s (10,6 s en frío). Tras Ctrl-C y rearranque, la conversación se precarga en 0,10 s y el siguiente turno reutiliza 8.141 tokens.
  - 166.561 tokens: guardar 0,51 s, restaurar 0,10 s (objetivo ≤ 2 s); el turno siguiente reutiliza 166.590 de 166.598 tokens y tarda 2,4 s en total, frente a 317 s en frío. Ojo: el archivo se acababa de escribir y la lectura salió de la caché de páginas del sistema; la lectura en frío del SSD no está medida (requiere `sudo purge`), aunque unos 3,5 GB caben con holgura en 2 s a la velocidad del SSD del M4 Max.
- `/metrics` añade `disk_cache_sessions`, `disk_cache_bytes`, `disk_loads_total`, `disk_saves_total` y los tiempos del último guardado y la última carga.
- Pendiente: medir el desgaste del SSD con el uso diario (pregunta abierta).
- Revisión (subagente), corregido:
  - Una conversación cargada del disco ya no se borra: su archivo queda como respaldo hasta que otra versión la sustituye. Si una petición falla, la sesión vuelve a su última instantánea en vez de perderse. Al leer de disco, solo se borra un archivo si es incompatible.
  - `--aligned-prefix` usa solo instantáneas a las que se llegó con trozos completos desde el principio (marca guardada también en disco) y nunca el final vivo, así que es bit-exacto de verdad.
  - Las sesiones vacías no se guardan ni se precargan.
  - Bifurcar recorta la capacidad al prefijo (la primera escritura copia solo eso), y solo se bifurca si la rama original tiene otro turno de usuario; regenerar una respuesta sustituye la anterior.
  - Cierre: Ctrl-C o SIGTERM rechazan la cola, paran la petición en curso (que se conserva, C-7), guardan y salen; una segunda señal sale sin guardar.
  - Al cargar se valida que cada caché tenga exactamente los tokens guardados; al guardar se hace `fsync` del archivo y del directorio y se borra el temporal si algo falla.
  - El modelo se identifica por un hash de `config.json`, el índice de tensores y los tamaños de los archivos, no por el nombre del directorio; en modo alineado el manifiesto incluye el tamaño del trozo.

**Hito 7: drafter DSpark (estado)**

- Port operación a operación del drafter de mlx-vlm (a su vez de vLLM): 3 capas con fusión DFlare de los estados del modelo principal en las pasadas 5, 23 y 41 y la salida final, y cabeza Markov recurrente (proyección plegada en el vocabulario). Propone hasta 7 tokens por bloque.
- El contexto del drafter vive en la sesión: se alimenta trozo a trozo durante el prefill, sigue a cada token aceptado y entra en las instantáneas y en el disco, así que una conversación recuperada de la caché sigue especulando sin recalcular.
- Verificación: un forward del modelo principal sobre el último token y los borradores; se aceptan los que coinciden con el greedy (o, con temperatura, cada borrador `d` con probabilidad p(d) y, si no, se muestrea de p sin `d`, lo que reproduce exactamente la distribución del modelo). Las cachés se rebobinan a lo aceptado: las globales por offset, la ventana quitando las últimas claves y los estados de convolución desde sus ventanas. Un token de parada entre los aceptados corta la ronda ahí.
- Longitud adaptativa (M-7): la política de mlx-vlm para DFlash según la aceptación reciente (entre 4 y 8), y además según la velocidad medida con y sin borrador: si especular no compensa, decodifica sin borrador y vuelve a probar cada 16 decisiones.
- La verificación de 2 a 8 tokens con caché bf16 usa el kernel de decode directo (que ahora también lee bf16): con el kernel de prefill era lenta a contexto largo. Prueba K-8 añadida.
- Test: con greedy, la especulación coincide con el decode normal hasta un casi-empate (la primera diferencia, en el token 155 de 200, con 0,17 nats entre los dos mejores en bf16 y 0,11 en 8 bits), en bf16 y 8 bits.
- Medido (greedy, caché 16 bits salvo indicación):
  - Contexto corto (código Python, 400 tokens): 45,8 tok/s con drafter frente a 34,3 sin él; aceptación 52 %.
  - 31k de contexto (500 tokens): 31,1 tok/s con drafter frente a 32,5 sin él; aceptación 41 %. El objetivo de ≥ 40 tok/s a 29k no se cumple. Con contexto largo verificar un bloque cuesta casi lo que gana: el MoE lee los expertos de hasta 8 tokens y la atención hace 8 veces el trabajo por clave. Queda pendiente un kernel de verificación con multiplicación matricial (tipo flash, repartido por claves), que encaja con K-2 del Hito 8.
- `--no-draft` lo desactiva y `--draft-block` limita el bloque (la política elige entre 2, 4 y 6 sin pasar de ese límite). `/metrics` añade `draft_rounds_total`, `drafted_tokens_total` y `accepted_tokens_total`.
- Revisión (subagente), corregido:
  - La ruta de verificación bf16 la tomaban también los trozos de prefill de 2 a 8 tokens, lo que rompía la paridad con la referencia; ahora solo la verificación (test de paridad con un prompt de 2.055 tokens).
  - Con `--aligned-prefix` no se especula (sus decisiones dependen de tiempos medidos y no son reproducibles).
  - El drafter ya no forma parte del manifiesto en disco: cambiar `--no-draft` no borra las conversaciones guardadas; una sesión guardada sin estado del drafter sigue sin especular.
  - El kernel de prefill empaquetado recibe buffers contiguos; el plegado de la cabeza Markov se hace en bloques de 4.096 filas (sin 1,85 GB temporales).
- Coste de una ronda (`bench_speculative_round`, caché bf16):

  | Contexto | Decode | Borrador (7) | Ronda bloque 2 | Ronda bloque 4 | Ronda bloque 8 |
  |---|---|---|---|---|---|
  | 1k | 28,9 ms | 12,4 ms | 65,9 ms → 2,0 tok | 70,9 ms → 3,4 tok | 91,4 ms → 2,9 tok |
  | 31k | 31,7 ms | 8,3 ms | 66,3 ms → 1,8 tok | 78,6 ms → 3,1 tok | 187 ms → 3,5 tok |

  El bloque de 8 nunca compensa; con texto que el drafter acierta, el de 4 da ~39 tok/s a 31k.
- Política de bloque (versión final): elige entre 1 (sin borrador), 2, 4 y 6 por tokens por segundo medidos en ventanas de 8 rondas, descartando la primera de cada tamaño, con un margen del 5 % para cambiar y sondeos cada 24 decisiones. Se reinicia en cada petición: compartida entre peticiones arrastraba lo aprendido a otro contexto (con 31k aprendía «no especules» y luego, a contexto corto, apenas volvía a probar).
  - Contexto corto (código): 46,4 tok/s frente a 34,3 sin drafter (aceptación 51 %).
  - 31k (ensayo, aceptación 39 %): 31,0 frente a 32,5; con poca aceptación lo mejor es casi no especular y la exploración cuesta ~5 %.
- Instantáneas periódicas: como mucho las 8 más recientes por sesión. Cada una retiene las 40 ventanas tal como quedan tras un trozo de prefill (hasta ~0,5 GB), así que un prompt de 512k habría acumulado unos 32 GB.

**Hito 8: prefill sobre la caché cuantizada (estado)**

- K-2 y K-3: con caché de 8 o 4 bits, el prefill ya no descomprime el historial de ninguna pasada global. Una variante del kernel relativo carga cada bloque de claves y valores directamente del formato empaquetado, con la misma expresión de descompresión que `mx.dequantize`, y su salida es bit a bit la del kernel bf16 sobre la caché descomprimida (test con 4 y 8 bits, grupos 32/64/128, contextos de 40 a 3.000 y bloques parciales). Así que los resultados no cambian.
- Medido con 166.561 tokens y caché de 8 bits: pico transitorio máximo de 1,74 GB (antes 2,13); la velocidad no cambia (398 tok/s a 145k, 319 s en frío), porque la descompresión no era el cuello de botella.
- Bias sobre la marcha (parte de K-2): no se hace, a propósito. En esta arquitectura la tabla de bias mide L × 24 × 1.024 (unos 100 MB por pasada con trozos de 2.048) y no crece con el contexto, así que el problema de Ollama no existe; calcularla dentro del kernel cambiaría el orden de las sumas y rompería la paridad exacta con la referencia.
- Pendiente (deseable): K-6 (fusionar las operaciones pequeñas y `compile` por pasada) y K-7 (hecho más abajo, para pocos tokens). El kernel de verificación especulativa está hecho (siguiente apartado).

**Contexto de 512k y kernel de verificación (22 de noviembre de 2026)**

El uso previsto es con 512k de contexto, así que las pruebas largas se hacen con un prompt de aguja de 516.533 tokens y caché de 8 bits.

- Servidor de punta a punta (8 bits, drafter activo):
  - Prefill en frío: 2.108 s (247 tok/s de media). Pico de 38,2 GB, transitorio máximo de 2,97 GB y 35,5 GB activos al terminar. Encuentra la aguja.
  - Guardado en disco: 4,1 s (11,5 GB). Al reiniciar se precarga en 2 s. El turno siguiente reutiliza 516.652 de 516.664 tokens.
  - El primer token tras cargar una sesión de disco tarda ~1 s: MLX lee los pesos de la sesión al usarlos por primera vez.
  - En disco solo se guarda la instantánea del final del último prompt: reenviar un turno anterior de la misma conversación vuelve a hacer el prefill completo. Es lo esperado (un chat siempre extiende la conversación), pero conviene saberlo al hacer pruebas.
- Decode sin especular a 516k: 51 ms por token (19,6 tok/s); antes 56. La atención global es la mitad (8 pasadas de ~3 ms) y el resto, sobre todo el MoE, cuesta lo mismo que a contexto corto.
- Lo que hacía lenta la especulación a 512k (6,7 tok/s en el servidor) era la atención con varias consultas:
  - El kernel de decode por carriles guarda en registros las G × L filas de consulta y se desborda a partir de 2 consultas. Con 8 bits a 516k, por pasada: 3,6 ms con 1 consulta, 16,7 con 3 y 168 con 7.
  - En el drafter, su capa global concatenaba toda su caché bf16 (copia de 2 GB, 10 ms) y usaba la atención de MLX con 8 consultas (19 ms).
- Kernel nuevo (`quant_verify_source`, hoy repartido en `VERIFY_HEAD`, un preludio de claves y `VERIFY_BODY`): cada simdgroup lleva 8 filas de consulta sobre una parte de las claves de una cabeza KV. Lee claves y valores de la caché directamente a registros de matriz, descomprimidos como `mx.dequantize`, y multiplica con matrices 8×8 en float. El softmax en línea se hace sobre los propios registros. Sin memoria de grupo ni barreras: la versión que montaba bloques en memoria de grupo era 2–4 veces más lenta. La pasada de reducción es la del kernel de decode.
  - Hace falta desenrollar del todo los bucles sobre la dimensión de la cabeza; un índice no constante en los arrays de matrices los manda a memoria y es 4–5 veces más lento.
  - Tiene un modo bidireccional sin bias ni sumidero para el bloque del drafter: sus 8 claves se escriben tras el contexto en el propio buffer de la caché y se descartan al terminar. El historial no se copia salvo cuando el buffer tiene que crecer o lo comparte otra sesión (tras un fork), copias que el siguiente `add_context` haría de todos modos.
  - Por pasada a 516k (`bench_speculative_attention`):

    | Consultas | 8 bits, por carriles | 8 bits, matrices | bf16, matrices |
    |---|---|---|---|
    | 1 | 3,6 ms | 2,9 ms | 4,9 ms |
    | 2 | 13,4 ms | 3,0 ms | 4,7 ms |
    | 3 | 16,7 ms | 4,6 ms | — |
    | 5 | 40,5 ms | 4,7 ms | — |
    | 7–8 | 168 ms (7) | 6,6 ms (7) | 7,1 ms (8) |

  - Se usa para 2 o más consultas y, con una sola, desde 256k claves (a 160k empatan; con contextos cortos el de carriles es algo más rápido). Comprobado frente a la referencia en float64: 4 y 8 bits, grupos 32/64/128, de 1 a 8 consultas, y el modo bidireccional.
- Ronda a 516k (`bench_long` sobre la sesión guardada; texto arbitrario, así que la aceptación no es representativa): decode 51 ms, borrador 13 ms, ronda de bloque 2 en 75 ms → 1,8 tokens (42 ms por token), bloque 4 en 97 ms → 2,0 y bloque 6 en 121 ms → 1,3. Antes: 189, 284 y 632 ms.
- Servidor a 516k con los kernels nuevos (conversación guardada en disco; ensayo de 400 tokens sobre la aguja):
  - Con drafter: 20,8 tok/s, aceptación del 46 % (174 de 380).
  - Sin drafter: 17,4 tok/s, incluido el ~1 s del primer token tras cargar de disco.
  - No es exactamente la misma comparación: en greedy, los dos caminos se separaron en el primer token por un casi empate (uno empezó el ensayo directamente y el otro razonando antes), así que generaron textos distintos.
  - Prefill en frío: 239 tok/s.
  - Una sesión guardada por un servidor con `--no-draft` no trae estado del drafter y, al cargarla con drafter, no especula hasta la siguiente conversación.
- Desglose de una ronda de bloque 2 a 1k: borrador 4 ms, construir el grafo 3 ms, pasada de verificación 37 ms (frente a 26 de un decode) y rebobinado 1 ms. Casi todo lo que cuesta de más la verificación es el MoE: dos tokens leen hasta el doble de expertos.

**Comparación con Ollama y MLX `main` (22 de noviembre de 2026)**

- Misma prueba que el borrador de PR para Ollama en M4 Max: prompt de aguja de ~29k tokens (31.157 con nuestra plantilla), razonamiento «low», temperatura 0, un nonce distinto por petición (sin caché de prefijos) y caché bf16. Respuesta correcta en todas.

  | | Ollama (MLX 0.32.4-82, kernels M4 Max) | Calamardo |
  |---|---:|---:|
  | Prefill | 570 tok/s | 782–803 tok/s |
  | Generación con drafter | 36,7 tok/s (34,1–40,8) | 35,9 tok/s (30,5–40,7), aceptación 49–54 % |
  | Generación sin drafter | — | 32,5 tok/s |

  Ollama generó 742–1.600 tokens por respuesta y Calamardo 117–201. En respuestas tan cortas la política de bloque pasa buena parte del tiempo explorando.
- Los kernels de Inkling que añadió Ollama (`nvfp4_matvec`, `nvfp4_matmul`, activaciones NVFP4) son suyos, no de MLX. El de expertos para pocos tokens, que lee cada experto una vez para todos los tokens que lo eligen, solo se activa en M5; en M4 usa el `gather_qmm` normal de MLX, como nosotros.
- MLX `main` (82 commits tras 0.32.4, sin rueda publicada) mejora `quantized_matmul` con pocas filas (`qmv_mma`), pero no `gather_qmm`. En este modelo solo están cuantizados los expertos; las capas densas son bf16. Actualizar exigiría compilar MLX desde el código (`cmake`; sin compilador `metal`, en modo JIT), y la ganancia esperable es pequeña: menos coste de CPU por llamada a kernels propios y algún arreglo, como la cuantización afín de grupos con todos los valores negativos.
- Lo que más falta para pasar de ~36 tok/s: un kernel de expertos para verificación que lea cada experto una vez para todos los tokens que lo eligen (K-7, siguiente apartado).

**K-7: expertos de pocos tokens leídos una vez (22 de noviembre de 2026)**

- Dónde se va una pasada de verificación (caché de 8 bits, contexto de 1k):

  | Tokens | Forward | Solo expertos (48 pasadas) |
  |---|---|---|
  | 1 | 30,3 ms | 7,9 ms |
  | 2 | 40,8 ms | 12,0 ms |
  | 4 | 48,9 ms | 19,4 ms |
  | 8 | 61,6 ms | 32,6 ms |

  - El `gather_qmm` de MLX hace un producto matriz-vector por ruta, así que un experto elegido por varios tokens se lee una vez por token: 4 tokens idénticos (los mismos 8 expertos) cuestan 18,1 ms frente a 7,9 de uno. No ayuda ninguna caché.
  - En rondas reales los tokens comparten expertos: de 16 rutas con 2 tokens, 13,5 expertos distintos; de 32 con 4, 21,1; de 64 con 8, 31,4.
- Kernel (`GATHER_QMV_SHARED_SOURCE`):
  - Un grupo de hilos por ruta y bloque de 8 filas de salida. La primera ruta de cada experto hace el trabajo de todas las que lo usan (como mucho una por token) y las demás salen; las rutas se encuentran con dos votos SIMD (64 rutas como máximo).
  - Cada bloque de pesos se descodifica una vez y se multiplica por las entradas de cada ruta, de una en una (con todas a la vez en registros se desbordaba).
  - Reproduce bit a bit el `fp_gather_qmv` de MLX 0.32.4 y su variante rápida, que MLX usa cuando K es múltiplo de 2 · 32 · 32 / bits (de nuestras formas, solo la proyección gate/up de los expertos mxfp8). Se replica el mismo orden de operaciones por carril, el bloque parcial final y que `scale * accum` vaya en su propia sentencia: en una sola expresión el compilador la fusiona en un FMA y redondea distinto.
  - Comprobado con pesos sintéticos en los tres formatos, nvfp4 con y sin escala global, mxfp4 y mxfp8, con bloques finales completos y parciales (sin códigos NaN en e4m3: un NaN hace que todo dé NaN y la comparación no pruebe nada) y con los expertos del checkpoint de cada tipo de capa.
- Se usa con 2 a 7 tokens (menos de 64 rutas). Desde 64 rutas MLX ordena las rutas y usa otro kernel que redondea distinto, así que ahí se queda `gather_qmm`; la verificación nunca pasa de 6 tokens. La paridad greedy con la referencia en Python sigue siendo exacta, incluido el prompt de 2.055 tokens, cuyo último trozo de prefill tiene 6.
- Resultado (expertos sobre las 48 pasadas):

  | Tokens | MLX | K-7 | Tokens idénticos: MLX → K-7 |
  |---|---|---|---|
  | 2 | 12,0 ms | 11,2 ms | 11,5 → 8,5 ms |
  | 4 | 19,4 ms | 17,8 ms | 18,1 → 10,5 ms |

  Forward de verificación con texto real: 2 tokens 38,9 ms (antes 40,8) y 4 tokens 44,3 ms (antes 48,9). Ronda de bloque 4 a 1k: 50,5 ms por 3,1 tokens (antes 53,8).
- Servidor con la prueba de Ollama (~29k, razonamiento «low», caché bf16, sin caché de prefijos): con drafter 34,7 / 42,9 / 41,5 tok/s (mediana 41,5; antes 35,9), con aceptación del 35–58 %; sin drafter 31,8 tok/s. Ollama hacía 36,7. Primer objetivo de ≥ 40 tok/s a 29k alcanzado en la mediana, aunque depende de la aceptación de cada respuesta.
- Lo siguiente con más peso: pasar de 1 a 2 tokens añade ~8 ms fuera de los expertos. Casi todo está en las 40 pasadas de ventana. Con más de un token siguen el camino de la referencia, que reordena la ventana y le concatena las claves nuevas: varias copias de ~4 MB por pasada, más un recorte en el siguiente decode. Las proyecciones densas bf16 apenas cuentan (menos de 1 ms). Resuelto en el apartado siguiente.

**Ventanas deslizantes en la verificación (22 de noviembre de 2026)**

- En la verificación (2 a 8 tokens), las 40 pasadas de ventana ya no tocan su caché. Las claves nuevas esperan aparte (`PassCache.pending`) y el kernel de verificación con matrices lee el búfer circular tal como está más esas claves. La posición de cada hueco se deduce del índice de escritura, sin copiar la ventana a orden temporal. Al rebobinar, las claves aceptadas se escriben una a una en el búfer, como lo haría un decode; las rechazadas no dejan rastro.
- El kernel comparte el cuerpo con el de la caché global. Solo cambia un preludio que da la fila y la posición de cada clave, y se añade la máscara de ventana. Comprobado frente a float64 con la ventana llena y rotada, a medio llenar y más larga que la ventana en orden temporal (tras un trozo de prefill), de 1 a 8 consultas.
- El decode normal y el prefill no cambian: la paridad con la referencia sigue exacta.
- Forward de verificación (caché de 8 bits, contexto de 1k): 2 tokens 32,8 ms (antes 38,9; 40,8 antes de K-7) y 4 tokens 39,0 ms (antes 44,3). Pasar de 1 a 2 tokens ya cuesta 2,4 ms más que un decode, en vez de 10.
- Rondas (`bench_speculative_round`, caché bf16):

  | Contexto | Decode | Bloque 2 | Bloque 4 | Bloque 8 |
  |---|---|---|---|---|
  | 1k | 28,9 ms | 38,5 ms → 1,9 tok | 45,5 ms → 3,1 tok | 64,5 ms → 3,9 tok |
  | 31k | 30,8 ms | 41,4 ms → 2,0 tok | 48,3 ms → 3,6 tok | 68,1 ms → 3,1 tok |

  Con K-7 y sin este cambio: 44,3, 50,5 y 69,2 ms a 1k; 47,4, 53,5 y 73,2 ms a 31k.
- Servidor con la prueba de Ollama (~29k, razonamiento «low», caché bf16): con drafter 39,2 / 52,9 / 51,5 tok/s (mediana 51,5; antes 41,5) con aceptación del 36–60 %; sin drafter 31,9. Ollama hacía 36,7.

**K-6: menos lanzamientos por token (22 de noviembre de 2026)**

- Perfil de un decode (1k de contexto, caché bf16; `bench_decode_profile`, hecho con un subagente):
  - 28,1 ms por token y **9.133 kernels de GPU**. Dos tercios son operaciones elementales sueltas; 2.692 son conversiones de tipo.
  - La CPU tarda 3 ms en construir el grafo y ~21 en codificarlo y enviarlo (unos 180 búferes de comandos). La GPU tarda 24,2 ms, unos 10 de ellos en latencia de lanzamientos pequeños.
  - El suelo por ancho de banda es de ~10 ms (unos 5,2 GB leídos por token) y, con las velocidades reales de los productos grandes, ~13 ms.
- Herramientas: MLX-C trae `mlx_export_to_dot` (el grafo perezoso, para contar nodos), `mlx_metal_start_capture` (requiere Xcode para leer la traza, no disponible) y `mlx_compile`. `ops::Compiled` envuelve este último: funciona desde Rust y da los mismos resultados bit a bit, porque cada operación conserva su tipo de salida. Solo fusiona cadenas elementales (normas, productos, concatenaciones y convoluciones quedan aparte). Se usa solo con 1 a 8 tokens, para no trazar de nuevo cada tamaño de trozo de prefill, y las posiciones nunca son constantes dentro de una función compilada.
- Hecho, todo exacto (tests bit a bit frente a las operaciones que sustituye y paridad greedy con la referencia en Python):
  - Máscara de la atención en decode: un kernel en vez de ~30 operaciones por pasada. Solo selecciona y copia valores bf16.
  - Convoluciones cortas estáticas en decode: un kernel que forma la ventana, hace la convolución depthwise como el `depthwise_conv_1d` de MLX (mismo orden por tap; los productos bf16 × bf16 son exactos en float), añade el residuo y escala. El estado siguiente sale de un solo kernel (`x · 1`, valores idénticos) en vez de `arange` + `take_along_axis`. `mx.copy` no sirve: comparte el búfer y mantendría viva toda la ventana (en prefill, ~2,6 GB retenidos).
  - Compilados: recortes L2, puertas de mensajes, activación de los expertos, combinación de expertos, puerta de la atención y la norma de v.
  - K-7 también para un solo token: idéntico a `gather_qmm` y algo más rápido.
- Resultado: 4.349 kernels por token (antes 9.133). Decode a 1k: 22,8 ms (antes 28,9); a 31k: 24,9 ms (antes 30,8). Tiempo de GPU por token: 21,6 ms (antes 24,2). Rondas a 31k: bloque 2 en 37,4 ms y bloque 4 en 44,0 ms (antes 41,4 y 48,3).
- Servidor con la prueba de Ollama (~29k, razonamiento «low», caché bf16): sin drafter 39,6 tok/s en las tres peticiones (antes 31,9; +24 %); con drafter 49,3 / 47,2 / 59,6 (mediana 49,3, aceptación del 42–55 %; con drafter la cifra depende sobre todo de la aceptación de cada respuesta).
- Ahora manda la GPU: la CPU ya no es el límite. Lo que queda:
  - Unos 4.000 kernels pequeños: conversiones de tipo, normas, sumas, copias y concatenaciones entre la atención y el MoE.
  - Expertos a ~285 GB/s frente a ~440 de las proyecciones densas.
  - Juntar las proyecciones con la misma entrada (q, r y puerta; también k y v en las ventanas sin mensajes). En decode es exacto: MLX elige la misma variante de `gemv` para esos tamaños, pero no para la de sumideros (N = 24), y en prefill cambia el tiling.
  - La atención en profundidad (~8 µs por llamada).

**Incoherencias a resolver**

1. Con un coste por token lineal en el contexto (solo crecen las 8 pasadas globales), 200 tok/s a 145k implican unos 9,5 min para 164k en frío. Para ≤ 7 min hacen falta unos 300 tok/s a 145k. Resuelto en la práctica: Calamardo hace 401 tok/s a 145k y 5,3 min en frío.
2. C-1 con instantáneas completas cada 8.192 tokens acumula unos 30–40 GB por conversación de 160k. La caché global solo crece por el final: los bloques sellados deben compartirse entre instantáneas, y solo copiarse las ventanas (unos 168 MB). Matiza el requisito 6 de la caché KV.
3. El puerto por defecto 11434 choca con Ollama durante las comparativas (verificación 6).
4. Restaurar 160k desde disco: unos 3,6 GB a 8 bits con drafter y ventanas. Cabe en 2 s solo con lectura directa (mmap) a los buffers.
