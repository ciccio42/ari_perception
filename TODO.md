# TODO: comunicazione tra Jetson e spark

Stato al 6 novembre 2026. La spark è la macchina `10.68.0.131`, dove gira `stt_server`.

## Problema

- **Whisper:** con FastDDS su tutta la Jetson la richiesta arriva ad `asr_node`, che trascrive, ma la risposta non compare sulla spark.
- **Prestazioni:** con FastDDS `/obj` oscilla tra 3 e 10 Hz e `RcvbufErrors` cresce molto più che con Cyclone. Il legame tra i due fatti non è dimostrato.
- **CycloneDDS:** dalla Jetson raggiunge i nodi della spark all'indirizzo sbagliato. Prova a usare `172.17.0.1`, che è il `docker0` della spark ma anche quello della Jetson, quindi quei nodi non compaiono.

## Regola da rispettare

Tra FastDDS e Cyclone i topic funzionano, i servizi e le action no.

| Server | Client | Risposta |
|---|---|---|
| FastDDS | FastDDS | Funziona |
| Cyclone | Cyclone | Funziona |
| Cyclone | FastDDS | Nessuna risposta |
| FastDDS | Cyclone | Nessuna risposta |

## Da fare

### 1. Provare i buffer di rete

È il test più economico e potrebbe bastare a stabilizzare `/obj`.

- [ ] Sulla Jetson, eseguire:
  ```bash
  sudo sysctl -w net.core.rmem_max=26214400 net.core.rmem_default=26214400
  ```
- [ ] Riavviare `demo_launch`.
- [ ] Misurare `/obj` per alcuni minuti e controllare se `RcvbufErrors` smette di crescere.
- [ ] Se funziona, rendere l'impostazione permanente.

### 2. Verificare la risposta di `/asr/whisper`

Il controllo è valido con qualsiasi opzione in cui `asr_node` resta su FastDDS.

- [ ] Riavviare `stt_server` sulla spark.
- [ ] Sulla spark, eseguire:
  ```bash
  timeout 15 ros2 service call /asr/whisper messages/srv/Asr "{audio_features: {data: [0.0]}}"; echo exit=$?
  ```
- [ ] Cercare `client will not receive response` nel terminale di `demo_launch`.

| Risultato | Significato | Azione |
|---|---|---|
| Stampa `text=''` | La risposta arriva alla spark | Controllare il codice di `stt_server` |
| `exit=124` | La risposta non torna dalla Jetson alla spark | Limitare FastDDS ai soli indirizzi LAN su entrambe le macchine |

### 3. Passare alla configurazione ibrida

Da fare se FastDDS resta troppo lento anche con i buffer aumentati.

| Nodi | RMW |
|---|---|
| `asr_node` | FastDDS |
| Altri nodi sulla Jetson | CycloneDDS |
| Nodi sulla spark | FastDDS, limitato a `10.68.0.131` |

- [ ] In `ai_nodes.launch.py`, aggiungere ad `asr_node`:
  ```python
  additional_env={'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp'},
  ```
- [ ] Togliere `-e RMW_IMPLEMENTATION=rmw_fastrtps_cpp` dal `docker run` nel README.
- [ ] Sulla spark, creare un profilo FastDDS (bozza non ancora provata):
  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>lan_only</transport_id>
        <type>UDPv4</type>
        <interfaceWhiteList>
          <address>10.68.0.131</address>
        </interfaceWhiteList>
      </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="lan_only_participant" is_default_profile="true">
      <rtps>
        <userTransports>
          <transport_id>lan_only</transport_id>
        </userTransports>
        <useBuiltinTransports>false</useBuiltinTransports>
      </rtps>
    </participant>
  </profiles>
  ```
- [ ] Impostare `FASTRTPS_DEFAULT_PROFILES_FILE` su quel file e riavviare tutti i nodi della spark.
- [ ] Rifare una build pulita sulla Jetson.

Controlli dopo il passaggio:

- [ ] I nodi della spark sono visibili dalla Jetson.
- [ ] I topic di percezione arrivano alla spark: `softbio_node` e `asd_data_node` devono ricevere dati. La lettura di topic Cyclone dalla spark non è ancora stata provata.
- [ ] Una frase pronunciata al robot torna come testo a `stt_server`.
- [ ] `/obj`, `/models_data_sync/data` e `/view/image/compressed` restano stabili intorno a 10 Hz.

**Da non fare:** spostare tutta la spark su Cyclone. Le action e i servizi verso il robot, che è su FastDDS, smetterebbero di funzionare.

## Dopo la scelta

- [ ] Aggiornare `docs/models_report.md`, che oggi riporta misure fatte con CycloneDDS.
- [ ] Aggiungere al README la nota sulla build pulita quando si cambia RMW.
- [ ] Decidere se tenere in `whisper_huggie.py` la lettura che ignora `layout`. La gestione degli errori nella callback va mantenuta.
