
# ARI Perception: report modelli

Misure del 6 novembre 2026 su `demo_launch` completo, con la camera reale di ARI.

## Sintesi

- Sono in uso 8 modelli: 4 con TensorRT, 2 con ONNX Runtime CUDA, 1 su CPU e 1 con esecuzione non verificata.
- Nessun modello è quantizzato: la riduzione di precisione usata è FP16.
- Il sistema regge il carico (CPU al 66%, GPU in media al 45%), ma ha poca RAM libera: 1.2 GB su 30.6 GB.
- La memoria dipende dal runtime più che dai modelli: ogni nodo AI occupa oltre 3 GB, anche con un modello da 1.2 MB.
- Le latenze di SixDRepNet, ResNet50 MTN, Whisper e Silero non sono ancora misurate.

## Come vengono eseguiti i nodi

**I nodi vengono lanciati tramite SSH da un'altra macchina.** Il container e tutti i nodi girano sulla Jetson di ARI: la macchina remota apre solo la sessione. CPU, GPU e memoria riportate qui sono quindi quelle della Jetson. Le latenze non includono la rete.

| Voce | Valore |
|---|---|
| Hardware | Jetson AGX Orin, 8 core, 30.6 GB di RAM condivisa con la GPU |
| Sistema | L4T r35, JetPack 5.1.1, profilo energetico `MODE_40W` |
| Container | `pepper_demo:r35` |
| Runtime | ONNX Runtime 1.16.1, TensorRT 8.5.2, torch 2.0.0, MediaPipe 0.10.9 |

## Modelli e ottimizzazione

| Modello | Nodo | Funzione | File su disco | Precisione su disco | Ottimizzazione | Esecuzione |
|---|---|---|---|---|---|---|
| UltraFace RFB-320 | `face_detection_node` | Rilevamento volti | ONNX, 1.2 MB | FP32 | Conversione FP16 all'avvio, engine TensorRT da 1 MB | GPU, TensorRT |
| Light-ASD | `asd_node` | Rilevamento di chi parla | ONNX, 4.1 MB | FP32 | Conversione FP16 all'avvio, senza TensorRT | GPU, CUDA |
| SixDRepNet | `gaze_estimation_node` | Direzione della testa | ONNX, 157 MB | FP32 | Conversione FP16 all'avvio, engine TensorRT da 79 MB | GPU, TensorRT |
| ResNet50 MTN | `soft_biometrics_node` | Età, genere, emozione, etnia | ONNX, 94 MB | FP32 | Grafo ONNX FP32, engine FP16 compilato da TensorRT, 48 MB | GPU, TensorRT |
| YOLOv8s | `obj_detector_node` | Rilevamento oggetti | ONNX, 45 MB | FP32 | Conversione FP16 all'avvio, engine TensorRT da 25 MB | GPU, TensorRT |
| Whisper large-v3-turbo | `asr_node` | Trascrizione vocale | ONNX, 2.2 GB (3 grafi) | FP16 | Ottimizzazione Optimum per GPU | CUDA richiesto, alcune operazioni su CPU |
| Silero VAD | `asr_node` | Rilevamento del parlato | ONNX, 2.3 MB | non verificata | Nessuna | Non verificata |
| MediaPipe Hands | `image_hand_node`, `image_hand_fingers_node` | Mani per i giochi | TFLite, incluso nel pacchetto | n/d | XNNPACK | CPU |

Tutti i nodi TensorRT mantengono su disco l'ONNX: serve all'avvio per creare la sessione e ritrovare l'engine in cache. La prima compilazione di un engine può richiedere molti minuti (circa 19 per YOLOv8s, circa 9.5 per UltraFace). Con la cache, YOLOv8s si carica in circa 17 s.

## Tempi di inferenza

Misurati a parte su singoli modelli, con il sistema scarico. Sotto il carico completo i tempi possono essere più alti.

| Modello | Tempo | Note |
|---|---|---|
| UltraFace RFB-320 | circa 11 ms | Solo modello |
| Light-ASD | circa 11 ms | Solo modello; circa 38 ms con la pipeline completa; 194 ms su CPU |
| YOLOv8s | circa 27 ms | Include il preprocessing |
| MediaPipe Hands | circa 47 ms | Su CPU |
| SixDRepNet | non misurato | Il nodo mantiene 10 Hz |
| ResNet50 MTN | non misurato | Il nodo mantiene 10 Hz |
| Whisper large-v3-turbo | non misurato | Il servizio è attivo, ma durante le misure non ha trascritto audio |
| Silero VAD | non misurato | |

## Carico per nodo

Medie su 3 finestre da 15 s con il launch completo in esecuzione. Il 100% corrisponde a 1 core. La memoria è la RSS del processo.

| Nodo | Modello | CPU media | Intervallo CPU | Memoria | GPU | Frequenza |
|---|---|---|---|---|---|---|
| `face_detection_node` | UltraFace | 92% | 20–131% | 3.3 GB | Sì | 10.0 Hz |
| `image_ari_input_node` | nessuno | 86% | 78–91% | 42 MB | No | 12.9 Hz |
| `obj_detector_node` | YOLOv8s | 43% | 43% | 4.0 GB | Sì | 9.1 Hz |
| `view_node` | nessuno | 36% | 36–37% | 29 MB | No | 9.7 Hz |
| `asd_node` | Light-ASD | 18% | 9–26% | 3.3 GB | Sì | 20.0 Hz (vedi sotto) |
| `model_data_sync_node` | nessuno | 14% | 14% | 22 MB | No | 9.9 Hz |
| `gaze_estimation_node` | SixDRepNet | 11% | 5–17% | 3.5 GB | Sì | 10.0 Hz |
| `soft_biometrics_node` | ResNet50 MTN | 6% | 5–8% | 3.4 GB | Sì | 10.1 Hz |
| `image_hand_node` | MediaPipe | sotto 2% | n/d | 87 MB | No | non misurata |
| `image_hand_fingers_node` | MediaPipe | sotto 2% | n/d | 83 MB | No | non misurata |
| `asr_node` | Whisper, Silero | sotto 2% (a riposo) | n/d | 3.4 GB | Sì | a richiesta |

- **GPU per nodo:** la colonna indica solo se il nodo usa la GPU. Il sistema espone il carico complessivo, non quello del singolo processo.
- **`asr_node`:** il carico vale a riposo, quando il nodo non sta trascrivendo.
- **Nodi dei giochi:** in questo campione usavano 83–87 MB. In un test separato, con MediaPipe in uso, circa 350 MB.
- **`/asd/speaking`:** i 20 Hz non sono un valore reale. In `asd_node.py` manca un `return` e ogni messaggio viene pubblicato due volte.

## Carico complessivo

| Risorsa | Valore |
|---|---|
| CPU | 54% utente, 9% sistema, 34% inattiva; load average circa 4.8 su 8 core |
| GPU | media 45%, mediana 39%, 95° percentile 74%, massimo 98% |
| Temperatura | CPU 43 °C, GPU 39 °C |
| RAM | 28.9 GB usati su 30.6 GB, 1.2 GB disponibili |
| Swap | 1.3 GB usati |
| Camera | 13.1 Hz in ingresso |

### Come leggere questi valori

- **Memoria:** RAM e memoria GPU coincidono, quindi non esiste un valore separato per la GPU. Le RSS dei nodi non vanno sommate, perché includono librerie condivise.
- **Costo del runtime:** gran parte della memoria è runtime e non modello. In un test separato, ogni nodo AI ha occupato circa 2.3 GB tra contesto CUDA e sessione TensorRT, a prescindere dalle dimensioni del modello.
- **CPU e GPU totali:** comprendono anche ciò che gira fuori dai nodi, come il desktop e VS Code (circa 90% di un core) e lo strumento di misura.
- **GPU a riposo:** in una misura precedente, a nodi fermi, ho letto 46–58%. Il valore è da ricontrollare, quindi il contributo netto dei nodi alla GPU non è ancora stabilito.

## Limiti delle misure

- Il campione copre circa un minuto. Non è stato misurato un periodo lungo né una conversazione attiva.
- Il carico di `face_detection_node`, `asd_node` e `gaze_estimation_node` varia molto tra le finestre. Non ho verificato se dipenda dai volti inquadrati.
- Durante il campionamento i comandi `ros2 topic hz` hanno aggiunto circa il 10% di un core.
- L'esecuzione reale di Silero e la quota di operazioni Whisper su CPU non sono state verificate.

## Prossimi passi

1. Misurare le latenze mancanti a launch fermo.
2. Misurare `asr_node` durante una trascrizione reale.
3. Correggere la doppia pubblicazione di `/asd/speaking`.
4. Valutare come ridurre la memoria, per esempio con meno sessioni CUDA separate.
