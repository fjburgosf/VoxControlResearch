"""Centralised interface translations (ES / EN)."""
from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "app_subtitle": {"es": "Reconocimiento de intención por voz con incertidumbre calibrada (UCIL)",
                     "en": "Uncertainty-aware voice intent recognition (UCIL)"},
    "tutorial": {"es": "Tutorial", "en": "Tutorial"},
    "lang_toggle": {"es": "ES | EN", "en": "ES | EN"},
    "model_none": {"es": "Modelo: no entrenado", "en": "Model: not trained"},
    "model_ready": {"es": "Modelo listo · semilla {seed} · calibrador {cal} · OOD {ood}",
                    "en": "Model ready · seed {seed} · calibrator {cal} · OOD {ood}"},
    "busy": {"es": "Trabajando…", "en": "Working…"},
    # tabs
    "tab_home": {"es": "Inicio", "en": "Home"},
    "tab_text": {"es": "Probar orden", "en": "Text test"},
    "tab_voice": {"es": "Voz", "en": "Voice"},
    "tab_adapt": {"es": "Adaptación", "en": "Adaptation"},
    "tab_exp": {"es": "Experimentos", "en": "Experiments"},
    "tab_results": {"es": "Resultados", "en": "Results"},
    "tab_settings": {"es": "Configuración", "en": "Settings"},
    # home
    "home_intro": {
        "es": ("VoxControlResearch estudia cómo debe actuar un sistema de control por voz cuando una orden es clara, "
               "ambigua, desconocida o está afectada por errores de reconocimiento. Para cada orden estima la "
               "intención, su incertidumbre calibrada y el riesgo de la acción, y decide EJECUTAR, CONFIRMAR o "
               "RECHAZAR/ACLARAR. Todas las acciones ocurren en un escritorio simulado (sandbox)."),
        "en": ("VoxControlResearch studies how a voice-control system should act when a command is clear, "
               "ambiguous, unknown or affected by recognition errors. For each command it estimates the intent, "
               "its calibrated uncertainty and the action risk, and decides EXECUTE, CONFIRM or REJECT/CLARIFY. "
               "All actions happen on a simulated desktop (sandbox).")},
    "train_model": {"es": "Entrenar modelo", "en": "Train model"},
    "load_model": {"es": "Cargar modelo…", "en": "Load model…"},
    "save_model": {"es": "Guardar modelo…", "en": "Save model…"},
    "examples": {"es": "Ejemplos", "en": "Examples"},
    "run_example": {"es": "Ejecutar ejemplo", "en": "Run example"},
    # text tab
    "command": {"es": "Orden", "en": "Command"},
    "analyse": {"es": "Analizar", "en": "Analyse"},
    "user": {"es": "Usuario", "en": "User"},
    "active_app": {"es": "Aplicación activa", "en": "Active application"},
    "use_context": {"es": "Usar contexto", "en": "Use context"},
    "f_intent": {"es": "Intención", "en": "Intent"},
    "f_slots": {"es": "Slots", "en": "Slots"},
    "f_raw": {"es": "Confianza bruta  p_max", "en": "Raw confidence  p_max"},
    "f_cal": {"es": "P(correcta) calibrada", "en": "Calibrated P(correct)"},
    "f_pin": {"es": "P(dentro del dominio)", "en": "P(in-domain)"},
    "f_ood": {"es": "Puntaje OOD  z", "en": "OOD score  z"},
    "f_risk": {"es": "Riesgo de la acción", "en": "Action risk"},
    "f_decision": {"es": "Decisión", "en": "Decision"},
    "f_options": {"es": "Opciones", "en": "Options"},
    "factors": {"es": "Factores reales de la decisión", "en": "Actual decision factors"},
    "factor": {"es": "Factor", "en": "Factor"},
    "value": {"es": "Valor", "en": "Value"},
    "confirm_exec": {"es": "Confirmar y ejecutar en sandbox", "en": "Confirm and run in sandbox"},
    "choice": {"es": "Opción a ejecutar:", "en": "Option to run:"},
    "correct_as": {"es": "Corregir como:", "en": "Correct as:"},
    "apply_correction": {"es": "Aplicar corrección", "en": "Apply correction"},
    "sandbox_state": {"es": "Estado del escritorio simulado", "en": "Simulated desktop state"},
    "reset_sandbox": {"es": "Reiniciar sandbox", "en": "Reset sandbox"},
    "EXECUTE": {"es": "▶ EJECUTAR", "en": "▶ EXECUTE"},
    "CONFIRM": {"es": "? CONFIRMAR", "en": "? CONFIRM"},
    "REJECT": {"es": "✕ RECHAZAR / ACLARAR", "en": "✕ REJECT / CLARIFY"},
    "corrected": {"es": "Corrección guardada para el usuario '{user}'. Se conserva al cerrar el programa y al "
                        "reentrenar.",
                  "en": "Correction stored for user '{user}'. It is kept after closing the program and after "
                        "retraining."},
    "executed": {"es": "Sandbox: {msg}", "en": "Sandbox: {msg}"},
    "not_executed": {"es": "Sandbox: {msg}", "en": "Sandbox: {msg}"},
    # voice tab
    "record": {"es": "Grabar", "en": "Record"},
    "seconds": {"es": "Duración  t [s]", "en": "Duration  t [s]"},
    "open_audio": {"es": "Abrir audio (WAV/FLAC)…", "en": "Open audio (WAV/FLAC)…"},
    "asr_model": {"es": "Modelo ASR", "en": "ASR model"},
    "asr_lang": {"es": "Idioma del comando", "en": "Command language"},
    "asr_missing": {"es": "Modelo de reconocimiento no disponible.\n{err}\n\nInstale faster-whisper "
                          "(pip install faster-whisper) o use el modo texto.",
                    "en": "Speech recognition model not available.\n{err}\n\nInstall faster-whisper "
                          "(pip install faster-whisper) or use text mode."},
    "transcript": {"es": "Transcripción", "en": "Transcript"},
    "asr_unc": {"es": "Incertidumbre ASR  U_ASR", "en": "ASR uncertainty  U_ASR"},
    # adaptation
    "corrections": {"es": "Correcciones guardadas (M_user)", "en": "Stored corrections (M_user)"},
    "export_csv": {"es": "Exportar CSV…", "en": "Export CSV…"},
    "clear_corrections": {"es": "Borrar todas las correcciones…", "en": "Delete all corrections…"},
    "clear_confirm": {"es": "¿Borrar todas las correcciones guardadas? No se puede deshacer.",
                      "en": "Delete every stored correction? This cannot be undone."},
    "col_user": {"es": "Usuario", "en": "User"},
    "col_text": {"es": "Transcripción", "en": "Transcript"},
    "col_wrong": {"es": "Predicha", "en": "Predicted"},
    "col_correct": {"es": "Correcta", "en": "Correct"},
    "col_unc": {"es": "Incertidumbre", "en": "Uncertainty"},
    # experiments
    "config": {"es": "Configuración", "en": "Configuration"},
    "seeds": {"es": "Semillas (separadas por espacio)", "en": "Seeds (space separated)"},
    "run": {"es": "Ejecutar experimento", "en": "Run experiment"},
    "exp_done": {"es": "Terminado: {id}", "en": "Finished: {id}"},
    "n_runs": {"es": "{n} corrida(s)", "en": "{n} run(s)"},
    "exp_note": {"es": "Cada experimento usa únicamente los parámetros de su archivo YAML (costos, predictor, "
                       "calibrador, detector, umbrales y semillas). La pestaña «Configuración» solo cambia el "
                       "modelo interactivo de «Probar orden» y «Voz». Al elegir un archivo se cargan sus semillas.",
                 "en": "Each experiment uses only the parameters of its YAML file (costs, predictor, calibrator, "
                       "detector, thresholds and seeds). The “Settings” tab only changes the interactive model of "
                       "“Text test” and “Voice”. Choosing a file loads its seeds."},
    "one_seed": {"es": "Aviso: con una sola semilla no se estiman la DE ni el IC 95 % (aparecen como NA). Para "
                       "reproducir las tablas del manual conserve todas las semillas del archivo.",
                 "en": "Note: with a single seed the SD and the 95% CI are not estimable (shown as NA). To "
                       "reproduce the tables of the manual keep every seed of the file."},
    "not_estimable": {"es": "NA", "en": "NA"},
    "exp_desc_main": {"es": "E1/E2/E4/E9: reconocimiento, paráfrasis, OOD y riesgo",
                      "en": "E1/E2/E4/E9: recognition, paraphrase, OOD and risk"},
    # results
    "experiment": {"es": "Experimento", "en": "Experiment"},
    "table": {"es": "Tabla", "en": "Table"},
    "figure": {"es": "Figura", "en": "Figure"},
    "refresh": {"es": "Actualizar", "en": "Refresh"},
    "open_folder": {"es": "Abrir carpeta", "en": "Open folder"},
    "export_zip": {"es": "Exportar (ZIP)…", "en": "Export (ZIP)…"},
    "no_results": {"es": "Aún no hay experimentos en la carpeta de resultados.",
                   "en": "No experiments in the results folder yet."},
    # settings
    "costs": {"es": "Costos de decisión (unidades de costo, u)", "en": "Decision costs (cost units, u)"},
    "p_cerr_low": {"es": "Costo de error, riesgo bajo  C_err,low [u]", "en": "Error cost, low risk  C_err,low [u]"},
    "p_cerr_medium": {"es": "Costo de error, riesgo medio  C_err,med [u]",
                      "en": "Error cost, medium risk  C_err,med [u]"},
    "p_cerr_high": {"es": "Costo de error, riesgo alto  C_err,high [u]", "en": "Error cost, high risk  C_err,high [u]"},
    "p_confirm": {"es": "Costo de confirmar  C_confirm [u]", "en": "Confirmation cost  C_confirm [u]"},
    "p_reject": {"es": "Costo de rechazar  C_reject [u]", "en": "Rejection cost  C_reject [u]"},
    "model_params": {"es": "Modelo", "en": "Model"},
    "p_seed": {"es": "Semilla aleatoria  s [–]", "en": "Random seed  s [–]"},
    "p_predictor": {"es": "Predictor de intención", "en": "Intent predictor"},
    "p_calibrator": {"es": "Calibrador", "en": "Calibrator"},
    "p_ood": {"es": "Detector OOD", "en": "OOD detector"},
    "p_mem": {"es": "Umbral de memoria  τ_mem [similitud coseno]", "en": "Memory threshold  τ_mem [cosine similarity]"},
    "p_context": {"es": "Usar el contexto (prior por aplicación activa)",
                  "en": "Use context (prior by active application)"},
    "exec_mode": {"es": "Modo de ejecución", "en": "Execution mode"},
    "mode_sandbox": {"es": "Todas las acciones se ejecutan en el escritorio simulado (sandbox). No hay ejecución "
                           "sobre el sistema operativo.",
                     "en": "Every action runs on the simulated desktop (sandbox). Nothing is executed on the "
                           "operating system."},
    "mandatory": {"es": "Confirmar siempre acciones de riesgo alto (también en el sandbox)",
                  "en": "Always confirm high-risk actions (also in the sandbox)"},
    "load_warning": {"es": "Cargue solo modelos creados por usted con este programa. Un archivo de modelo (.pkl) "
                           "de origen desconocido puede ejecutar código en su equipo. ¿Continuar?",
                     "en": "Only load models you created with this program. A model file (.pkl) of unknown "
                           "origin can run code on your computer. Continue?"},
    "apply_retrain": {"es": "Aplicar y reentrenar", "en": "Apply and retrain"},
    "thresholds": {"es": "Umbral mínimo de ejecución directa  p* [–]:  bajo {low:.3f} · medio {medium:.3f} · alto {high:.3f}",
                   "en": "Minimum direct-execution threshold  p* [–]:  low {low:.3f} · medium {medium:.3f} · high {high:.3f}"},
    "error": {"es": "Error", "en": "Error"},
    "need_model": {"es": "Primero entrene o cargue un modelo.", "en": "Train or load a model first."},
    # tutorial
    "tut_title": {"es": "Tutorial interactivo", "en": "Interactive tutorial"},
    "tut_do": {"es": "Hacerlo por mí", "en": "Do it for me"},
    "tut_next": {"es": "Siguiente ›", "en": "Next ›"},
    "tut_back": {"es": "‹ Anterior", "en": "‹ Back"},
    "tut_close": {"es": "Cerrar", "en": "Close"},
    "tut_step": {"es": "Paso {i} de {n}", "en": "Step {i} of {n}"},
    "tut_done": {"es": "✓ Paso completado", "en": "✓ Step completed"},
    "tut_pending": {"es": "Pendiente: realice la acción o pulse «Hacerlo por mí».",
                    "en": "Pending: perform the action or press “Do it for me”."},
}

FACTORS: dict[str, dict[str, str]] = {
    "top_intent_prob": {"es": "Probabilidad de la intención principal  p(1)", "en": "Top intent probability  p(1)"},
    "second_intent_prob": {"es": "Probabilidad de la segunda intención  p(2)", "en": "Second intent probability  p(2)"},
    "entropy": {"es": "Entropía normalizada  H/log K", "en": "Normalised entropy  H/log K"},
    "ood_score_z": {"es": "Puntaje OOD  z", "en": "OOD score  z"},
    "ood_detector": {"es": "Detector OOD", "en": "OOD detector"},
    "asr_uncertainty": {"es": "Incertidumbre ASR  U_ASR", "en": "ASR uncertainty  U_ASR"},
    "context_shift": {"es": "Cambio por contexto  U_context", "en": "Context shift  U_context"},
    "memory_weight": {"es": "Peso de la memoria del usuario  λ", "en": "User-memory weight  λ"},
    "action_risk": {"es": "Riesgo de la acción", "en": "Action risk"},
    "slot_status": {"es": "Estado de los slots", "en": "Slot status"},
    "expected_cost_execute": {"es": "Costo esperado de ejecutar  E[C|x] [u]", "en": "Expected cost of executing  E[C|x] [u]"},
    "expected_cost_confirm": {"es": "Costo esperado de confirmar  E[C|x] [u]", "en": "Expected cost of confirming  E[C|x] [u]"},
    "expected_cost_reject": {"es": "Costo esperado de rechazar  E[C|x] [u]", "en": "Expected cost of rejecting  E[C|x] [u]"},
    "mandatory_confirmation": {"es": "Confirmación obligatoria (riesgo alto)", "en": "Mandatory confirmation (high risk)"},
    "transcript": {"es": "Transcripción", "en": "Transcript"},
    "asr_confidence": {"es": "Confianza ASR", "en": "ASR confidence"},
}

SANDBOX: dict[str, str] = {
    "invalid": "acción no válida: {reason}", "opened": "se abrió {app}", "not_open": "{app} no está abierta",
    "closed": "se cerró {app}", "closed_lost": "se cerró {app} (se perdieron cambios sin guardar)",
    "switched": "ventana activa: {app}", "volume": "volumen {value}", "muted": "audio silenciado",
    "playing": "reproduciendo", "nothing_playing": "no hay nada reproduciéndose", "paused": "reproducción en pausa",
    "searched": "búsqueda: '{query}'", "no_text_target": "la ventana activa no admite texto",
    "typed": "texto escrito en {app}", "scroll": "desplazamiento {value}", "no_document": "no hay un documento activo",
    "saved": "se guardó {app}", "sent": "mensaje enviado a {contact}", "file_not_found": "no se encontró {file}",
    "trashed": "{file} se movió a la papelera", "screenshot": "captura de pantalla tomada",
    "needs_confirmation": "no se ejecutó: la decisión requiere confirmación explícita",
    "needs_choice": "no se ejecutó: elija primero una de las opciones ({options})",
    "unresolved_slot": "no se ejecutó: falta resolver {slots}",
    "rejected": "no se ejecutó: la petición fue rechazada",
}

VALUES: dict[str, dict[str, str]] = {
    "low": {"es": "bajo", "en": "low"}, "medium": {"es": "medio", "en": "medium"},
    "high": {"es": "alto", "en": "high"}, "ok": {"es": "completos", "en": "complete"},
    "ambiguous": {"es": "ambiguo", "en": "ambiguous"}, "missing": {"es": "faltante", "en": "missing"},
    "invalid": {"es": "inválido", "en": "invalid"}, "fused": {"es": "fusión aprendida", "en": "learned fusion"},
    "fusion": {"es": "fusión", "en": "fusion"},
}

TERMS_ES: dict[str, str] = {
    # metrics
    "intent_accuracy": "exactitud de intención", "incorrect_execution_rate": "tasa de ejecuciones incorrectas",
    "correct_execution_rate": "tasa de ejecuciones correctas", "clarification_rate": "tasa de confirmación",
    "rejection_rate": "tasa de rechazo", "coverage": "cobertura", "selective_risk": "riesgo selectivo",
    "false_rejection_rate": "tasa de rechazos falsos", "ood_execution_rate": "tasa de ejecución de OOD",
    "ood_rejection_rate": "tasa de rechazo de OOD",
    "high_risk_incorrect_execution_rate": "ejecuciones incorrectas de riesgo alto",
    "wrong_execution_cost": "costo de ejecuciones erróneas", "mean_cost": "costo medio",
    "burden_confirmations": "carga: confirmaciones", "burden_repeats": "carga: repeticiones",
    "burden_corrections": "carga: correcciones", "user_burden_per_command": "carga del usuario por orden",
    "accuracy": "exactitud", "macro_f1": "macro-F1", "macro_precision": "precisión macro",
    "macro_recall": "exhaustividad macro", "ece": "ECE", "mce": "MCE", "brier": "Brier", "nll": "NLL",
    "multiclass_nll": "NLL multiclase", "temperature": "temperatura", "auroc": "AUROC", "auprc": "AUPRC",
    "fpr95": "FPR al 95 % TPR", "aurc": "AURC", "oracle_aurc": "AURC oráculo", "risk_at_50": "riesgo al 50 %",
    "risk_at_80": "riesgo al 80 %", "acc_new_expressions": "exactitud en expresiones nuevas",
    "acc_old_knowledge": "exactitud en lo ya aprendido", "forgetting": "olvido", "adaptation_gain": "ganancia de adaptación",
    "user_accuracy": "exactitud del usuario", "global_accuracy": "exactitud global",
    "other_users_accuracy": "exactitud de otros usuarios", "wer": "WER", "mean_asr_uncertainty": "U_ASR media",
    "intent_error": "error del clasificador con texto de referencia (sin ASR)", "asr_induced_error": "error añadido por el ASR", "correct": "correctas",
    "mean_entropy": "entropía media", "mean_p_correct": "P(correcta) media",
    "asks_when_target_ambiguous": "pregunta ante destino ambiguo", "n": "n",
    # methods and strategies
    "B1_rules": "B1 reglas", "B2_argmax": "B2 argmax", "B3_fixed_threshold": "B3 umbral fijo",
    "B4_calibrated_threshold": "B4 umbral calibrado", "UCIL_context": "UCIL con contexto",
    "UCIL_no_incremental": "UCIL sin adaptación", "UCIL_noise_calibrated": "UCIL con calibración ante ruido",
    "UCIL_full": "UCIL completo", "no_ood": "sin OOD", "no_calibration": "sin calibración",
    "no_risk_routing": "sin enrutamiento por riesgo", "no_incremental": "sin aprendizaje incremental",
    "none": "ninguna", "memory": "memoria", "prototype": "prototipos", "finetune_naive": "ajuste ingenuo",
    "replay_fifo": "replay FIFO", "replay_class_balanced": "replay balanceado", "replay_uncertainty": "replay por incertidumbre",
    "replay_diversity": "replay por diversidad", "ucil_memory": "memoria UCIL",
    "clean": "limpio", "combined_shift": "cambio combinado", "in_domain": "en dominio",
    "in_domain_plus_ood": "en dominio + OOD", "near": "cercano", "far": "lejano", "all": "todos",
    "low": "bajo", "medium": "medio", "high": "alto",
}

COLUMNS: dict[str, dict[str, str]] = {
    "method": {"es": "método", "en": "method"}, "metric": {"es": "métrica", "en": "metric"},
    "mean": {"es": "media", "en": "mean"}, "median": {"es": "mediana", "en": "median"},
    "sd": {"es": "DE", "en": "SD"}, "q1": {"es": "Q1", "en": "Q1"}, "q3": {"es": "Q3", "en": "Q3"},
    "min": {"es": "mín", "en": "min"}, "max": {"es": "máx", "en": "max"},
    "ci95_low": {"es": "IC95 inf", "en": "95% CI low"}, "ci95_high": {"es": "IC95 sup", "en": "95% CI high"},
    "seed": {"es": "semilla", "en": "seed"}, "level": {"es": "nivel", "en": "level"},
    "model": {"es": "modelo", "en": "model"}, "strategy": {"es": "estrategia", "en": "strategy"},
    "n_corrections": {"es": "n correcciones", "en": "n corrections"}, "subset": {"es": "subconjunto", "en": "subset"},
    "detector": {"es": "detector", "en": "detector"}, "risk": {"es": "riesgo", "en": "risk"},
    "noise_rate": {"es": "tasa de ruido", "en": "noise rate"}, "setting": {"es": "escenario", "en": "setting"},
    "user": {"es": "usuario", "en": "user"}, "calibrator": {"es": "calibrador", "en": "calibrator"},
    "population": {"es": "población", "en": "population"},
}

TUTORIAL: list[dict] = [
    {"title": {"es": "Bienvenida", "en": "Welcome"},
     "body": {"es": "Este tutorial recorre un experimento completo: entrenar el modelo, analizar órdenes claras, "
                    "ambiguas y desconocidas, corregir al sistema, ejecutar un experimento y leer sus resultados. "
                    "En cada paso puede hacerlo usted mismo o pulsar «Hacerlo por mí».",
              "en": "This tutorial walks through a complete experiment: train the model, analyse clear, ambiguous "
                    "and unknown commands, correct the system, run an experiment and read its results. In each "
                    "step you can act yourself or press “Do it for me”."},
     "action": None, "check": None, "tab": "home"},
    {"title": {"es": "1 · Entrenar el modelo", "en": "1 · Train the model"},
     "body": {"es": "En «Inicio» pulse «Entrenar modelo». Se genera un corpus sintético reproducible (plantillas "
                    "por nivel de dificultad), se entrena el predictor, se calibra con un conjunto separado y se "
                    "selecciona el detector OOD. La barra superior muestra el calibrador elegido.",
              "en": "On “Home” press “Train model”. A reproducible synthetic corpus is generated (templates by "
                    "difficulty level), the predictor is trained, calibrated on a separate split, and the OOD "
                    "detector is selected. The top bar shows the chosen calibrator."},
     "action": "train", "check": "model", "tab": "home"},
    {"title": {"es": "2 · Una orden clara", "en": "2 · A clear command"},
     "body": {"es": "En «Probar orden» escriba «abre spotify» y pulse «Analizar». Observe: la intención, la "
                    "confianza bruta, la probabilidad calibrada de acierto y la decisión EJECUTAR. La tabla de "
                    "factores muestra los valores reales que produjeron la decisión.",
              "en": "On “Text test” type “abre spotify” and press “Analyse”. Note the intent, the raw confidence, "
                    "the calibrated probability of being correct and the EXECUTE decision. The factor table shows "
                    "the actual values behind the decision."},
     "action": ("analyse", "abre spotify"), "check": ("analysed", "abre spotify"), "tab": "text"},
    {"title": {"es": "3 · Un destino ambiguo", "en": "3 · An ambiguous target"},
     "body": {"es": "Analice «cierra el chat». Hay dos aplicaciones de chat: el sistema no adivina, CONFIRMA y "
                    "ofrece las opciones. Además, cerrar es una acción de riesgo alto. Para ejecutarla elija la "
                    "aplicación en «Opción a ejecutar» y pulse «Confirmar y ejecutar en sandbox». Sin elegir una "
                    "opción no se ejecuta nada.",
              "en": "Analyse “cierra el chat”. Two chat applications match: the system does not guess, it "
                    "CONFIRMS and offers the options. Closing is also a high-risk action. To run it choose the "
                    "application under “Option to run” and press “Confirm and run in sandbox”. Nothing runs "
                    "until an option is chosen."},
     "action": ("analyse", "cierra el chat"), "check": ("analysed", "cierra el chat"), "tab": "text"},
    {"title": {"es": "4 · Una petición desconocida (OOD)", "en": "4 · An unknown request (OOD)"},
     "body": {"es": "Analice «pide un taxi al aeropuerto». No pertenece a ninguna intención registrada: el puntaje "
                    "OOD sube, P(dentro del dominio) baja y el sistema no la ejecuta. Aquí la RECHAZA. Con otras "
                    "peticiones fuera de dominio puede pedir confirmación en lugar de rechazar, pero no fuerza la "
                    "intención más parecida.",
              "en": "Analyse “pide un taxi al aeropuerto”. It matches no registered intent: the OOD score rises, "
                    "P(in-domain) drops and the system does not execute it. Here it REJECTS. With other "
                    "out-of-domain requests it may ask for confirmation instead of rejecting, but it does not force "
                    "the closest intent."},
     "action": ("analyse", "pide un taxi al aeropuerto"),
     "check": ("analysed", "pide un taxi al aeropuerto"), "tab": "text"},
    {"title": {"es": "5 · Riesgo de la acción", "en": "5 · Action risk"},
     "body": {"es": "En «Configuración» vea los costos C_err por nivel de riesgo y el umbral mínimo p* que "
                    "resulta para cada uno. Con la misma confianza, una acción de riesgo alto (borrar, enviar) "
                    "requiere más certeza para ejecutarse directamente. Ejecute el ejemplo 06 para verlo.",
              "en": "On “Settings” see the error costs C_err per risk level and the resulting minimum threshold p*. "
                    "With the same confidence, a high-risk action (delete, send) needs more certainty to run "
                    "directly. Run example 06 to see it."},
     "action": ("example", "example_06_risk_routing"), "check": ("example", "example_06_risk_routing"),
     "tab": "home"},
    {"title": {"es": "6 · Corregir al sistema", "en": "6 · Correct the system"},
     "body": {"es": "Analice «cállalo», elija «mute» en «Corregir como» y pulse «Aplicar corrección». Vuelva a "
                    "analizar: la memoria del usuario (M_user) cambia la predicción sin reentrenar ni modificar el "
                    "modelo global. La corrección aparece en «Adaptación».",
              "en": "Analyse “cállalo”, choose “mute” under “Correct as” and press “Apply correction”. Analyse "
                    "again: the user memory (M_user) changes the prediction without retraining or modifying the "
                    "global model. The correction appears under “Adaptation”."},
     "action": ("correct", "cállalo", "mute"), "check": "corrected", "tab": "text"},
    {"title": {"es": "7 · Ejecutar un experimento", "en": "7 · Run an experiment"},
     "body": {"es": "En «Experimentos» elija «exp_main.yaml»: aparecen sus diez semillas (100 a 109). Para una "
                    "prueba rápida deje solo una (por ejemplo 100), sabiendo que con una semilla no se estima el "
                    "IC 95 %. Pulse «Ejecutar experimento». Se comparan B1 reglas, B2 argmax, B3 umbral fijo, B4 "
                    "umbral calibrado y UCIL. Cada corrida recibe un identificador EXP-AAAA-NNNNNN.",
              "en": "On “Experiments” choose “exp_main.yaml”: its ten seeds appear (100 to 109). For a quick run "
                    "keep only one (e.g. 100), bearing in mind that a single seed gives no 95% CI. Press “Run "
                    "experiment”. B1 rules, B2 argmax, B3 fixed threshold, B4 calibrated threshold and UCIL are "
                    "compared. Each run gets an identifier EXP-YYYY-NNNNNN."},
     "action": ("experiment", "exp_main.yaml", "100"), "check": "experiment", "tab": "exp"},
    {"title": {"es": "8 · Leer los resultados", "en": "8 · Read the results"},
     "body": {"es": "En «Resultados» elija el experimento, la tabla «summary_decision_metrics» (media, DE e IC 95 % "
                    "por método) y la figura «fig_tradeoff_ier_vs_intervention»: tasa de ejecuciones incorrectas "
                    "frente a tasa de intervención, sin ponderar y ponderada por costo.",
              "en": "On “Results” choose the experiment, the table “summary_decision_metrics” (mean, SD and 95% CI "
                    "per method) and the figure “fig_tradeoff_ier_vs_intervention”: incorrect-execution rate versus "
                    "intervention rate, unweighted and cost-weighted."},
     "action": "results", "check": "results", "tab": "results"},
    {"title": {"es": "9 · Exportar", "en": "9 · Export"},
     "body": {"es": "«Exportar (ZIP)» empaqueta la carpeta del experimento: config.yaml, metadata.json (versión, "
                    "semillas, entorno), tablas CSV, figuras PNG/SVG/PDF y el registro. Con esa configuración "
                    "cualquiera puede repetir la corrida y obtener las mismas cifras.",
              "en": "“Export (ZIP)” packs the experiment folder: config.yaml, metadata.json (version, seeds, "
                    "environment), CSV tables, PNG/SVG/PDF figures and the log. With that configuration anyone can "
                    "repeat the run and obtain the same figures."},
     "action": None, "check": None, "tab": "results"},
]


class I18N:
    def __init__(self, lang: str = "es"):
        self.lang = lang
        self._bound: list[tuple] = []
        self.listeners: list = []

    def t(self, key: str, **kw) -> str:
        s = STRINGS.get(key, {}).get(self.lang) or STRINGS.get(key, {}).get("en") or key
        return s.format(**kw) if kw else s

    def bind(self, widget, key: str, option: str = "text"):
        self._bound.append((widget, key, option))
        widget.configure(**{option: self.t(key)})
        return widget

    def toggle(self) -> None:
        self.lang = "en" if self.lang == "es" else "es"
        for widget, key, option in self._bound:
            try:
                widget.configure(**{option: self.t(key)})
            except Exception:  # noqa: BLE001 - widget may have been destroyed
                pass
        for fn in self.listeners:
            fn()
