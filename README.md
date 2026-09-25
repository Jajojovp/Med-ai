# MEDAI — Historia de un proyecto de datos clínicos

### De 60 modelos con métricas brillantes a un sistema que sabe decir «no lo sé»

*Relato de análisis de datos. No contiene código. Cada cifra de este documento está medida sobre los ficheros del proyecto y se reproduce con los comandos del apartado 9.*

---

## Qué es MEDAI

MEDAI empezó como un MVP de machine learning en oncología. En lugar de reemplazar esos modelos, construí un sistema alrededor de ellos: una capa diseñada para que sus salidas sean trazables, reproducibles y explícitas sobre sus límites.

La arquitectura mantiene cuatro cosas separadas:

- **Model output** — lo que produce el artefacto entrenado
- **Dataset evidence** — lo que los datos subyacentes realmente pueden sostener
- **Validation evidence** — qué tan robusto es el resultado bajo distintas pruebas
- **Clinical interpretation** — qué se puede, y qué no se puede, afirmar médicamente

Esa separación es fundamental. Un modelo puede producir una predicción. Un dataset puede contener una señal fuerte. Un procedimiento de validación puede revelar qué tan estable es esa señal. Ninguna de esas cosas, por sí sola, establece validez clínica — así que MEDAI no las colapsa en un solo puntaje.

**Lo que incorpora el sistema:**

- 🩺 8 flujos clínicos, con varios modelos de machine learning operando dentro de cada flujo
- 🔬 Auditoría de modelos y datasets: particiones repetidas, control de duplicados, permutación de etiquetas y línea base de una sola variable
- 📊 La evidencia de validación se registra por separado de la salida del modelo, y solo se muestran valores medidos cuando la prueba correspondiente realmente se ejecutó
- 🗂️ Procedencia y estado del artefacto, distinguiendo verificación técnica de validación clínica
- ⚖️ Umbrales de decisión, desacuerdo entre modelos, valores faltantes e imputación quedan visibles en vez de ocultos detrás de una predicción final
- 🌐 Español, inglés y francés soportados tanto en la interfaz como en los análisis clínicos, desde una única fuente de verdad
- 🔬 Un modo de ensayo que permite probar el flujo clínico completo antes de contar con datos clínicos reales y contratos de variables verificados

Y una sola regla recorre toda la plataforma: **si la evidencia no existe, MEDAI no la fabrica.** Sin puntaje de validación externa inventado, sin imputación oculta, sin ponderación sin explicar, sin presentar rendimiento técnico del modelo como rendimiento clínico.

MEDAI es apoyo a la decisión, no un sistema de diagnóstico. El objetivo no es que un modelo de IA se vea impresionante: es que todo el camino datos → modelo → evidencia → interpretación sea inspeccionable.

Esta es la primera versión. La siguiente etapa es validación externa, contratos de variables verificados y datos reales para cada condición.

Lo que sigue es esa auditoría, contada con sus cifras.

---

## 0. Regla de lectura

Antes de cualquier número, tres reglas que gobiernan todo lo que sigue:

**Una cifra que ningún artefacto entrenado produjo no se muestra.** Si falta el dato, se declara la falta; no se rellena con ruido, ni con la media, ni con una constante. *(SPEC §19)*

> «200 particiones aleatorias demuestran estabilidad frente al split, pero no demuestran validez clínica ni ausencia de todo tipo de leakage.»

---

## 1. El porqué: una pregunta clínica, no una pregunta de machine learning

El razonamiento clínico es una secuencia: primero los síntomas, después los datos y la información de los exámenes, y con eso se confirma —o se descarta— la enfermedad. El médico no adivina con una analítica aislada: la usa para confirmar una sospecha que ya venía de la consulta.

De ahí salen dos exigencias que un modelo puede cumplir o no:

1. **El modelo confirma, no diagnostica.** Su lugar está en el tramo final de la secuencia, apoyando una decisión que sigue siendo del clínico.
2. **Para confirmar hacen falta datos de confirmación.** Si el modelo se entrena con respuestas a un cuestionario, lo más que puede aprender es a reproducir el cuestionario — no a confirmar una enfermedad.

Esta es la historia de cómo un conjunto de modelos que parecían funcionar extraordinariamente bien resultó no estar haciendo eso, y de lo que hizo falta para averiguarlo, medirlo y corregirlo.

---

## 2. La problemática: el punto de partida

El proyecto arrancó con una herencia voluminosa y, en apariencia, valiosa:

| Lo que había | Número |
|---|---|
| Artefactos de modelo (.pkl) en el catálogo | 60 |
| Enfermedades cubiertas por el catálogo | 12 |
| Scripts de generación auditados uno a uno | 79 |
| Artefactos con contrato clínico acreditable | **0** |

Cero de sesenta. Y el problema no era técnico: los 60 artefactos cargaban y ejecutaban sin errores. El problema era que ninguno podía acreditar de dónde salían sus datos ni qué etiqueta predecía realmente. Un modelo que dice AUC 0,98 y no puede mostrar el dataset que lo produjo no es evidencia clínica: es una cifra huérfana.

Tres comprobaciones hicieron visible el hueco:

- **Contrato clínico:** ningún artefacto declaraba esquema de variables, unidades, rangos, codificación ni exámenes requeridos. Se detectó y retiró además una bandera `schema_confirmed` que estaba puesta sin base medida.
- **Script de origen:** 60 de 60 artefactos con origen sintético atribuido y 0 con contrato propio; ninguno declaraba los nombres de sus variables. Y el único script que sí abría un dataset real —el de próstata— fabricaba su etiqueta con `load_breast_cancer()`, así que tampoco acreditaba. La regla que fijé: «Ningún PKL entra al catálogo clínico sin script de origen que abra un dataset real; y si el script fabrica los datos, el artefacto no acredita aunque su nombre diga wdbc» *(SPEC §16)*.
- **Cifras servidas:** la aplicación calculaba el «ensamble» en el navegador. Solo la regresión logística salía de coeficientes medidos; las otras cuatro probabilidades se generaban con ruido alrededor del valor logístico — la cifra que veía el médico no la había producido ningún modelo entrenado. Se sustituyó por una consulta real al motor de inferencia y, si el motor no responde, la app no estima: muestra solo la regresión logística, lo declara como tal y oculta la tabla de modelos y la insignia de consenso. De ahí nació la regla permanente §19.

---

## 3. Cómo lo encontré

### 3.1 El instrumento: cuatro capas que nunca se mezclan

El primer trabajo no fue de modelos, fue de arquitectura de la evidencia. Separé lo que se dice en cuatro capas y prohibí que una se apoyara en la de al lado:

```
Model Output          «qué probabilidad sale»        (el artefacto)
      ↓
Dataset Evidence      «de qué corpus sale»           (procedencia, etiqueta, nulos, licencia)
      ↓
Validation Evidence   «cuánto aguanta esa cifra»     (particiones, permutación, fuga, calibración)
      ↓
Clinical Interpretation «qué se puede y qué NO se puede afirmar con ella»
```

Sin esa separación, una cifra de un corpus sintético acaba presentada como rendimiento clínico en tres clics. Con ella, cada número arrastra su origen.

### 3.2 La pregunta que ordenó todo: ¿con qué datos se entrenó cada flujo?

Ordené los flujos según la naturaleza de sus datos, y el patrón apareció de inmediato:

| Flujo | Corpus | Tipo de dato | ¿Confirma? | AUC holdout medido |
|---|---|---|---|---|
| Mama | WDBC (FNA) | citología por punción | Sí | 0,9993 |
| Renal | UCI kidney | laboratorio (sangre/orina) | Sí | 1,0000 |
| Diabetes (Pima) | Pima Indians | OGTT + laboratorio | Sí | 0,8259 |
| Cardíaca | Cleveland | clínica + angiografía | Sí | 0,9481 |
| Hepático (ILPD) | UCI 225 ILPD | laboratorio (bilirrubina, enzimas) | Sí | 0,8388 |
| Diabetes temprana | cuestionario | síntomas autorreportados | No | 0,9988 |
| Pulmón | encuesta | 15 respuestas autorreportadas | No | 0,9421 |
| Hígado (legado) | UCI 423 HCC Survival | desenlace de supervivencia | No *(es pronóstico)* | 0,7615 |
| Cervical | conducta + biopsia | mixto, sin señal | No | 0,6155 |

El hallazgo: los flujos que se caen son, uno por uno, los que no usan datos de confirmación. Y los dos casos con AUC casi perfecto y datos no confirmatorios (early_diabetes 0,9988, pulmón 0,9421) demuestran que el AUC no sirve como criterio: mide qué tan bien ordena el modelo, no si lo que ordena tiene sentido clínico.

### 3.3 Los extremos: por qué 1,0000 y por qué 0,6155

Un AUC 1,0000 y un AUC 0,6 tenían que ser explicados, no presumidos ni escondidos. Los medí flujo a flujo con 200 particiones aleatorias, permutación de etiquetas, agrupación por duplicados y techo de una sola variable:

| Flujo | AUC | Veredicto medido | Lo que lo sostiene |
|---|---|---|---|
| Renal | 1,0000 | corpus casi separable por construcción | 200 particiones 0,9979 · 26,5 % dan exactamente 1,0 · la variable `sc` sola ya da 0,937 · 0 duplicados |
| Mama | 0,9993 | corpus casi separable | 200 particiones 0,9949 · `worst perimeter` sola da 0,9874 · 0 duplicados |
| Diabetes temprana | 0,9988 | fuga por filas repetidas + etiqueta circular | 72 de 104 filas de test con gemelo exacto · al deduplicar cae a 0,9735 → 0,9514 · Poliuria=1 & Polidipsia=1 → 193 de 193 positivos |
| Cardíaca | 0,9481 | señal real y estable | 0 duplicados · 0 % de particiones llegan a 1,0 · media agrupada 0,9000 · holdout 0,05 por encima de su propia media → la cifra citable es ~0,90 |
| Pulmón | 0,9421 | fuga leve + punto de operación inservible | 34 filas repetidas, 10 cruces train/test, 1 vector con etiqueta contradictoria · agrupar mueve −0,0019 (la fuga no es la causa) · el daño es el umbral: especificidad 0,75 |
| Diabetes (Pima) | 0,8259 | señal real moderada | — |
| Hepático (ILPD) | 0,8388 | señal real moderada | límites completos en §4.4 |
| Hígado (legado) | 0,7615 | etiqueta inadecuada | la hemoglobina sola (0,8654) supera al modelo completo · 200 particiones con mínimo 0,4654 · predice mortalidad a 1 año, no diagnóstico |
| Cervical | 0,6155 | corpus sin señal | permutación p=0,24 (nulo 0,4874) · 200 particiones media 0,5393 con mínimo 0,175 · Brier skill −1,2692 · la mejor de sus 23 variables sola da 0,5802 |

Tres conclusiones que cambian la lectura del proyecto entero:

1. **El 1,0000 no es sobreajuste ni fuga: es separabilidad del corpus.** El peligro no es que el modelo memorice, es que ese corpus no se parece a la clínica y el número no se transferirá.
2. **La inflación real estaba en early_diabetes**, el flujo con questionnaire: fuga por solape de filas más una etiqueta derivable de dos síntomas. Dos formas de circularidad, no una.
3. **El 0,6155 de cervical no es un modelo débil: es un corpus sin señal.** Con ese corpus, ningún algoritmo razonable va a funcionar; el fallo se corrige en el dato, no en el modelo.

### 3.4 La pregunta que el AUC esconde: ¿cuántos sanos marca?

Un AUC no responde la pregunta que el clínico se hace de verdad: «si le hago esto a 100 pacientes sin la enfermedad, a cuántos les voy a decir que sí». Lo medí en el umbral realmente servido, con cota superior exacta (Clopper-Pearson, 95 % unilateral):

| Flujo | Falsos positivos por 100 sanos | Cota superior 95 % | Enfermos que se escapan, por 100 |
|---|---|---|---|
| Renal | 0 (0 de 30) | 9,5 | 0 |
| Diabetes temprana | 2,5 (1 de 40) | 11,3 | 0 |
| Mama | 2,78 (2 de 72) | 8,5 | 0 |
| Cardíaca | 3,03 (1 de 33) | 13,6 | 14,3 |
| Cervical | 7,45 (12 de 161) | 11,8 | 81,8 |
| Hepático (ILPD) | 8,82 (3 de 34) | 21,3 | 38,5 |
| Pulmón | 25,0 (2 de 8) | 60,0 | 7,4 |
| Diabetes (Pima) | 29,0 (29 de 100) | 37,4 | 18,5 |
| Hígado (legado) | 50,0 (10 de 20) | 69,8 | 0 |

Dos detalles que el AUC no muestra y que el sistema de salud sí notaría: un 0 de falsos positivos en 30 sanos no es «0 %» — la cota dice que podría llegar hasta 9,5 por 100 — y un flujo puede marcar a la mitad de los sanos manteniendo un AUC decente.

Y la segunda pregunta, la que decide si la herramienta sirve fuera del holdout: a prevalencia realista (10 %), el valor predictivo positivo de pulmón cae a 0,29 y el del hígado legado a 0,18, frente al 0,96 y 0,57 que lucían en su propio holdout. La base rate no es un tecnicismo: es la diferencia entre una herramienta útil y una máquina de asustar sanos.

### 3.5 El sano con una analítica normal

Probado contra la aplicación en marcha, no en abstracto: el paciente sano típico pasa en los 9 flujos. Pero un sano con analíticas en el percentil 95 (valores altos, dentro de rango fisiológico) queda marcado como positivo en 7 de los 9 flujos. Esto es lo que hay que decirle a un evaluador clínico antes de que él lo descubra: la especificidad es el punto débil del conjunto, no el AUC.

### 3.6 Auditoría del auditor

La revisión no se detuvo en los modelos. En mi propia herramienta de auditoría aparecieron tres defectos, y los tres se arreglaron en el código, no en el texto del informe:

1. Un campo llamado «falsos positivos por 100 declarados» no medía falsos positivos por 100 sanos, sino la fracción falsa de los positivos declarados (100·FP/(TP+FP)): dos preguntas clínicas distintas. Afectaba a 8 de 9 flujos.
2. La «cota 95 %» que se publicaba era la regla de tres aplicada a ciegas: en cervical daba 1,86 cuando la tasa medida era 7,45 — un límite por debajo de lo observado, que es imposible. Se sustituyó por la cota exacta de Clopper-Pearson, que coincide con la regla de tres cuando no hay fallos (0 de 30 → 9,5 por 100) y es una cota real cuando los hay.
3. El índice de validación cubría 4 de 9 flujos y en esos cuatro la columna de validación cruzada venía vacía; 3 flujos no tenían fichero de procedencia.

---

## 4. Cómo planteo la solución

### 4.1 La arquitectura, en una frase

Una aplicación donde el clínico rellena el expediente de un paciente y recibe, por enfermedad, una probabilidad con su evidencia completa colgando debajo: de qué corpus sale, cómo se validó, qué límites tiene y qué no se puede afirmar con ella.

### 4.2 Las tres reglas que sostienen el sistema

| Regla | Qué impide |
|---|---|
| **§16** — script de origen real | que un artefacto con nombre de dataset famoso entre al catálogo sin abrir ese dataset |
| **§17** — techo declarado sin cohorte externa | que una validación interna se lea como validez clínica: no hay segunda cohorte → el techo se declara, no se rellena |
| **§19** — nada de cifras no producidas | que la interfaz muestre un número que ningún artefacto entrenado generó |
| **§20** — ensayo etiquetado | que un artefacto sin contrato se lea como clínico: los flujos de ensayo viajan con `schema_status: ensayo_sin_contrato` y la interfaz lo declara en cada corrida |

### 4.3 Lo que la aplicación se niega a hacer

No diagnostica, no recomienda tratamiento, no hace triaje autónomo y no muestra una probabilidad sin su ficha de procedencia. Cuando el dato falta, la app lo dice: la falta se declara.

### 4.4 La prueba de que el método cambia el resultado

El mejor ejemplo no es un flujo que siempre funcionó, sino uno que estaba mal y se reemplazó:

| Aspecto | Hígado legado | Hígado reemplazado (ILPD) |
|---|---|---|
| Corpus | UCI 423 HCC Survival | UCI 225 ILPD (DOI 10.24432/C5D02C) |
| Sujetos | 165 pacientes ya diagnosticados de HCC | 583 pacientes (441 H / 142 M) |
| Etiqueta | `died` — mortalidad a 1 año | diagnóstico de enfermedad hepática por marcadores bioquímicos |
| AUC holdout | 0,7615 | 0,8388 (IC 95 % 0,7704–0,9008) |
| ¿Supera a una sola variable? | No: hemoglobina sola 0,8654 | Sí: la mejor variable sola (`ast`) da 0,7539 |
| Falsos positivos por 100 sanos | 50 | 8,82 |
| Licencia | — | CC BY 4.0, verificada en la ficha primaria |

El flujo nuevo, medido a fondo:

- **Holdout:** 117 filas (83 con enfermedad) · AUC 0,8388 · AUPRC 0,9387 (IC 0,9105–0,9638) · Brier 0,1506 · Brier skill vs prevalencia 0,2694 · ECE 0,0876 · sensibilidad 0,6145 · precisión 0,9444 · matriz VP 51 · FP 3 · VN 31 · FN 32.
- **Estabilidad:** 200 particiones media 0,747 (mínimo 0,6258) · validación cruzada 5×3 con las tres semillas en 0,7507 / 0,7687 / 0,7487 · permutación de etiquetas p = 0 con nulo 0,5203.
- **El holdout es optimista, y se declara:** el propio informe de validación advierte que 0,8388 (holdout) frente a 0,7034 (predicciones fuera de fold) — la cifra citable es la baja, no la alta.
- **Límites sin maquillar:** no hay cohorte externa; 5 vectores idénticos cruzan train/test; 13 filas duplicadas (2,2 %); el umbral servido (0,72) se eligió sobre el mismo holdout; 32 de 83 enfermos no se detectan; el calibrador isotonic se midió (ΔBrier 0,0072) pero no se sirve con el artefacto; EPV efectivo 13,3 con 10 variables.
- **Un matiz de equidad ya medido:** en el holdout, el subgrupo de 90 casos con `gender=1` da 0,8738 y el de 27 casos con `gender=0` da 0,7434 — el rendimiento no es igual entre subgrupos y así está declarado.

---

## 5. Lo que hemos hecho (inventario verificable)

| Hecho | Estado verificado |
|---|---|
| Flujos servidos con corpus real y ficha | 9 |
| Autotest de la capa de evidencia | 42 de 42 en verde |
| Arnés de navegador, flujo hepático | 25 de 25 marcas |
| Arnés de control (mama) | 25 de 25 marcas |
| Enfermedades probadas en modo ensayo, en la app y en tres idiomas | 12 de 12 (5 modelos cada una, es/en/fr) |
| Licencias verificadas en la ficha primaria | 8 de 9 (CC BY 4.0); pulmón queda NO VERIFICADA, no rellenada |
| Limpieza de ficheros intermedios | 89,4 MB con manifiesto previo y comprobación sha256 |
| Informes de auditoría de extremos | 3 informes de agentes de ciencia de datos + adenda de verificación ítem por ítem del resultado |
| Defectos de herramienta corregidos | 3 (dos campos mal nombrados + una cota falsa) |

Y lo que no se hizo, dicho explícitamente: no se retiró del servicio ningún flujo reprobado sin sustituirlo, y hay un flujo (cervical) que sigue servido con AUC 0,6155 mientras se decide su retirada.

---

## 6. Lo que NO está resuelto

Sin suavizar, porque un evaluador lo va a encontrar de todas formas:

1. **No existe cohorte externa** para ningún flujo. Todo es validación interna sobre un único corpus. Es el techo declarado *(SPEC §17)*.
2. **5 vectores idénticos cruzan train/test** en el flujo hepático. Eliminarlos exige reentrenar; está medido y declarado, no oculto.
3. **Cervical sigue servido con AUC 0,6155** (p=0,24; 200 particiones con mínimo 0,175) y su valor almacenado (0,5957) no se reproduce desde sus cinco modelos: la discrepancia está en la regla de agregación, no en los miembros, y la regla no se ha localizado.
4. **El índice de validación cubre 4 de 9 flujos** y viene con la columna de validación cruzada vacía; 3 flujos sin fichero de procedencia.
5. **Pulmón se sirve con umbral 0,30**, que marca 25 de cada 100 sanos y tiene un beneficio clínico neto negativo (DCA −0,0230): avisar a todos sería mejor que usar el modelo.
6. **El hígado legado (HCC) sigue en el catálogo** con etiqueta de mortalidad; debería retirarse, no reetiquetarse.
7. **La fiabilidad de las probabilidades está medida; la utilidad clínica no**, porque no hay desenlace clínico ni cohorte externa.
8. **De los 334 artefactos heredados censados, 318 cargan y 16 no:** 12 por falta de `lightgbm` en este entorno y 4 por pickles antiguos que fallan al deserializar. El catálogo antiguo sigue pendiente de decisión.

---

## 7. En el lenguaje de un sistema de salud

Este proyecto, contado a un evaluador clínico o a un responsable digital, es esto:

- **Problem** — los equipos acumulan modelos con métricas internas brillantes y sin trazabilidad de datos ni de etiqueta; la consecuencia es que no se pueden llevar a una decisión clínica, aunque el número sea alto.
- **Requirements** — cada cifra debe poder rastrearse hasta su corpus, su etiqueta, su partición y su licencia; y el sistema debe declarar lo que no puede afirmar.
- **Workflow** — el clínico introduce síntomas y resultados de exámenes en un expediente por enfermedad; el sistema devuelve una probabilidad con su evidencia colgando debajo (procedencia, validación, límites) y explica el caso con un asistente guiado, en español, inglés y francés.
- **Solution** — capa de evidencia de cuatro niveles, catálogo de procedencia por flujo, motor de auditoría reproducible y una interfaz que se niega a mostrar cifras sin origen.
- **Decision Support** — el sistema no decide: ordena la información y expone la incertidumbre, incluida la que perjudica al propio sistema.

Lo que haría falta para pasar de prototipo a uso clínico: una cohorte externa prospectiva, revisión por comité de ética, registro del modelo con hash de código y fecha, y un plan de seguimiento de equidad por subgrupos. Nada de eso se afirma aquí como hecho.

---

## 8. Publicación: qué se publica y qué no

| Se publica | No se publica |
|---|---|
| Esta historia (narrativa y cifras) | Código fuente del backend y del frontend |
| Arquitectura y reglas (§16/§17/§20) | Artefactos entrenados (.pkl) |
| Cifras medidas y sus límites | Corpus descargados y ficheros intermedios |
| Procedencia y licencias de los corpus públicos | Credenciales, tokens o claves (no se observó ninguna) |
| Capturas de la interfaz con casos sintéticos | Datos de pacientes (no hay: los corpus son públicos y anonimizados) |

Los corpus citados son públicos y se usan bajo su licencia (CC BY 4.0, con atribución a UCI Machine Learning Repository). El código se mantiene privado: lo que se abre es la evidencia y el método, no la implementación.

---

## 9. Cómo verificarlo tú mismo

```bash
# Autotest de la capa de evidencia (42 comprobaciones)
PYTHONPATH=. ./.venv/Scripts/python.exe medai/tools/selftest.py

# Auditoría de métricas de los 9 flujos (200 particiones, permutación, fuga, cotas de FP)
PYTHONPATH=. ./.venv/Scripts/python.exe medai/tools/audit_metricas_flujos.py
```

Y los ficheros donde vive cada cifra:

| Cifra | Fichero |
|---|---|
| AUC, fuga, estabilidad, falsos positivos y cotas de los 9 flujos | `medai/reports/audit_metricas_8_flujos.json` |
| Validación del flujo hepático (holdout, CV, calibración, límites) | `medai/validation/reports/liver_ilpd.json` |
| Procedencia, licencia y unidades del corpus hepático | `data/liver_ilpd.provenance.json` |
| Integridad del corpus | `sha256 = 69ff1a69a3f3aca42f10691cb7dca0257242f56f605a47d2e0d51febaae37bb7` |
| Auditoría de extremos (propia y de agentes) | `medai/reports/extremos_medicion_propia.md`, `agente_extremos_altos_1.md`, `agente_extremos_altos_2.md`, `agente_extremos_bajos.md` |
| Reglas del sistema | `MEDAI-CLINICAL-AI-SPEC.md` (§16 admisión de artefactos, §17 cohorte externa, §19 probabilidad servida, §20 modo ensayo) |

---

## Apéndice · Glosario mínimo

- **AUC (AUROC):** probabilidad de que el modelo ordene un enfermo por encima de un sano. No es precisión, ni probabilidad, ni utilidad clínica.
- **AUPRC:** lo mismo pero centrado en la clase enferma; solo tiene sentido leído contra su valor trivial, que es la prevalencia.
- **Falsos positivos por 100 sanos:** de cada 100 pacientes sin la enfermedad, a cuántos marca como enfermos. Es la pregunta clínica del cribado.
- **Cota superior (Clopper-Pearson):** con pocos sanos, «0 fallos» no significa «0 %»; la cota dice hasta cuánto podría llegar la tasa real.
- **Prevalencia (base rate):** proporción de enfermos en la población. Bajarla destroza el valor predictivo de un test aunque su AUC no cambie.
- **Fuga (leakage):** información del conjunto de test que el modelo vio durante el entrenamiento (por ejemplo, filas duplicadas presentes en ambos).
- **Separabilidad del corpus:** el dataset separa las clases por su construcción. Un AUC de 1,0 ahí no mide calidad del modelo, mide el dataset.
- **EPV (eventos por variable):** cuántos casos de la clase minoritaria hay por cada variable predictora. Por debajo de 10, el modelo se apoya en el azar.
- **Brier score / ECE:** miden si la probabilidad que sale es la probabilidad que ocurre (fiabilidad), no solo si el orden es correcto.
- **DCA (análisis de decisión):** compara el beneficio neto del modelo contra «tratar a todos» y «no tratar a nadie». Un DCA negativo significa que usar el modelo es peor que no usarlo.

---

*Documento generado a partir de mediciones sobre el proyecto. Las cifras de esta versión corresponden a la auditoría del 2026-09-17; toda cifra que cambie debe volver a medirse antes de citarse.*
