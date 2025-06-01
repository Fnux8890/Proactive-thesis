# Architecture Diagrams

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Data Sources Layer"
        DS1[Greenhouse Sensors<br/>KnudJepsen & Aarslev]
        DS2[Weather API<br/>Open-Meteo]
        DS3[Energy API<br/>Danish Market]
        DS4[Literature DB<br/>Phenotypes]
    end
    
    subgraph "Ingestion Layer"
        IL1[Rust Async Pipeline<br/>10K rows/sec]
        IL2[Data Validators]
        IL3[Batch Writer]
    end
    
    subgraph "Storage Layer"
        DB[(TimescaleDB<br/>PostgreSQL 16)]
        subgraph "Hypertables"
            HT1[sensor_data]
            HT2[sparse_features]
            HT3[external_data]
        end
    end
    
    subgraph "Processing Layer"
        subgraph "Enhanced Sparse Pipeline"
            PL1[Hourly Aggregation<br/>Rust]
            PL2[Gap Filling<br/>Rust]
            PL3[GPU Features<br/>Python]
            PL4[Era Creation<br/>Rust]
        end
    end
    
    subgraph "ML Layer"
        ML1[LightGBM<br/>Growth Model]
        ML2[LightGBM<br/>Energy Model]
        ML3[LSTM<br/>Predictor]
    end
    
    subgraph "Optimization Layer"
        OL1[NSGA-III<br/>GPU Accelerated]
        OL2[Pareto Front<br/>Generator]
    end
    
    subgraph "Output Layer"
        OUT1[Control Strategies]
        OUT2[Dashboards]
        OUT3[API Endpoints]
    end
    
    DS1 --> IL1
    DS2 --> PL1
    DS3 --> PL1
    DS4 --> PL3
    
    IL1 --> IL2 --> IL3 --> DB
    DB --> PL1
    PL1 --> PL2 --> PL3 --> PL4
    PL4 --> DB
    
    DB --> ML1
    DB --> ML2
    DB --> ML3
    
    ML1 --> OL1
    ML2 --> OL1
    ML3 --> OL1
    
    OL1 --> OL2
    OL2 --> OUT1
    OL2 --> OUT2
    OL2 --> OUT3
    
    style PL3 fill:#f9b,stroke:#333,stroke-width:2px
    style OL1 fill:#f9b,stroke:#333,stroke-width:2px
    style DB fill:#99f,stroke:#333,stroke-width:2px
```

## Detailed Pipeline Flow

```mermaid
flowchart LR
    subgraph "Stage 1: Ingestion"
        A1[CSV Files] --> A2[Parser]
        A3[JSON Files] --> A2
        A2 --> A4[Validator]
        A4 --> A5[Batch Queue]
        A5 --> A6[Async Insert]
    end
    
    subgraph "Stages 2-4: Enhanced Processing"
        B1[Raw Data] --> B2{Coverage Check}
        B2 -->|>10%| B3[Process Window]
        B2 -->|<10%| B4[Skip Window]
        
        B3 --> B5[Aggregate Hourly]
        B5 --> B6{Gap Size?}
        B6 -->|≤2h| B7[Fill Gap]
        B6 -->|>2h| B8[Keep NULL]
        
        B7 --> B9[GPU Extract]
        B8 --> B9
        B9 --> B10[Create Era]
    end
    
    subgraph "Stage 5: Modeling"
        C1[Features] --> C2[Train/Val Split]
        C2 --> C3[LightGBM GPU]
        C3 --> C4[Model Validation]
        C4 --> C5[Save Models]
    end
    
    subgraph "Stage 6: Optimization"
        D1[Load Models] --> D2[Init Population]
        D2 --> D3[GPU Evaluate]
        D3 --> D4{Converged?}
        D4 -->|No| D5[Evolve]
        D5 --> D3
        D4 -->|Yes| D6[Pareto Front]
    end
    
    A6 --> B1
    B10 --> C1
    C5 --> D1
```

## Hybrid Rust-Python Architecture

```mermaid
sequenceDiagram
    participant R as Rust Orchestrator
    participant DB as TimescaleDB
    participant P as Python GPU Process
    participant G as GPU Hardware
    
    R->>DB: Query sensor data
    DB-->>R: Return sparse data
    
    R->>R: Aggregate hourly
    R->>R: Detect gaps
    R->>R: Create windows
    
    R->>P: Send window batch (JSON)
    P->>G: Allocate GPU memory
    G-->>P: Memory allocated
    
    P->>G: Launch feature kernels
    G-->>P: Compute features
    
    P->>P: Format results
    P-->>R: Return features (JSON)
    
    R->>DB: Store features
    DB-->>R: Confirm storage
    
    Note over R,G: Process repeats for each batch
```

## GPU Memory Management

```mermaid
graph TB
    subgraph "GPU Memory Layout"
        subgraph "Device Memory (8GB)"
            DM1[Feature Tensors<br/>2GB]
            DM2[Model Weights<br/>1GB]
            DM3[Working Memory<br/>4GB]
            DM4[Reserved<br/>1GB]
        end
        
        subgraph "Host Memory"
            HM1[Pinned Buffers<br/>512MB]
            HM2[Staging Area<br/>256MB]
        end
    end
    
    subgraph "Transfer Flow"
        T1[Host→Device DMA]
        T2[Kernel Execution]
        T3[Device→Host DMA]
    end
    
    HM1 --> T1
    T1 --> DM3
    DM3 --> T2
    T2 --> DM3
    DM3 --> T3
    T3 --> HM2
```

## Data Sparsity Handling

```mermaid
flowchart TB
    subgraph "Traditional Approach (Fails)"
        TA1[5-min Data] --> TA2[Time Regularization]
        TA2 --> TA3[91.3% NULLs]
        TA3 --> TA4[Interpolation]
        TA4 --> TA5[Synthetic Data]
        TA5 --> TA6[False Patterns]
        TA6 --> TA7[❌ Bad Models]
    end
    
    subgraph "Our Approach (Works)"
        OA1[5-min Data] --> OA2[Hourly Aggregation]
        OA2 --> OA3[Coverage Filter]
        OA3 --> OA4[Conservative Fill]
        OA4 --> OA5[Real Patterns]
        OA5 --> OA6[Sparse Features]
        OA6 --> OA7[✅ Good Models]
    end
    
    style TA7 fill:#f96
    style OA7 fill:#9f9
```

## Docker Service Dependencies

```mermaid
graph LR
    subgraph "Infrastructure"
        DB[(PostgreSQL)]
        CACHE[(Redis)]
    end
    
    subgraph "Pipeline Services"
        S1[rust_pipeline]
        S2[enhanced_sparse_pipeline]
        S3[model_builder]
        S4[moea_optimizer]
    end
    
    subgraph "Monitoring"
        M1[Prometheus]
        M2[Grafana]
        M3[pgAdmin]
    end
    
    DB --> S1
    S1 --> S2
    DB --> S2
    S2 --> S3
    DB --> S3
    S3 --> S4
    DB --> S4
    
    S1 -.-> M1
    S2 -.-> M1
    S3 -.-> M1
    S4 -.-> M1
    
    M1 --> M2
    DB -.-> M3
    
    style S2 fill:#bbf,stroke:#333,stroke-width:2px
    style S3 fill:#f9b,stroke:#333,stroke-width:2px
    style S4 fill:#f9b,stroke:#333,stroke-width:2px
```

## Feature Extraction Pipeline

```mermaid
graph TB
    subgraph "Input Window"
        IW[12-hour window<br/>73% coverage]
    end
    
    subgraph "CPU Features (Rust)"
        CF1[Coverage Ratio]
        CF2[Gap Statistics]
        CF3[Continuity Score]
        CF4[Island Detection]
    end
    
    subgraph "GPU Features (Python)"
        subgraph "Statistical"
            GF1[Mean/Std/Percentiles]
            GF2[Skewness/Kurtosis]
            GF3[Autocorrelation]
        end
        
        subgraph "Temporal"
            GF4[Trend Detection]
            GF5[Seasonality]
            GF6[Change Points]
        end
        
        subgraph "Domain"
            GF7[Thermal Time]
            GF8[VPD Calculation]
            GF9[DLI Accumulation]
        end
    end
    
    subgraph "Output"
        OUT[2000+ Features<br/>JSONB Format]
    end
    
    IW --> CF1
    IW --> CF2
    IW --> CF3
    IW --> CF4
    
    IW --> GF1
    IW --> GF2
    IW --> GF3
    IW --> GF4
    IW --> GF5
    IW --> GF6
    IW --> GF7
    IW --> GF8
    IW --> GF9
    
    CF1 --> OUT
    CF2 --> OUT
    CF3 --> OUT
    CF4 --> OUT
    GF1 --> OUT
    GF2 --> OUT
    GF3 --> OUT
    GF4 --> OUT
    GF5 --> OUT
    GF6 --> OUT
    GF7 --> OUT
    GF8 --> OUT
    GF9 --> OUT
    
    style GF1 fill:#f9b
    style GF2 fill:#f9b
    style GF3 fill:#f9b
    style GF4 fill:#f9b
    style GF5 fill:#f9b
    style GF6 fill:#f9b
    style GF7 fill:#f9b
    style GF8 fill:#f9b
    style GF9 fill:#f9b
```

## MOEA Optimization Flow

```mermaid
graph TB
    subgraph "Initialization"
        I1[Random Population<br/>100 individuals]
        I2[Control Variables<br/>24 hourly setpoints]
    end
    
    subgraph "GPU Evaluation"
        E1[Batch to GPU]
        E2[Extract Features]
        E3[Model Predictions]
        E4[Calculate Objectives]
        
        subgraph "Objectives"
            O1[Growth Rate]
            O2[Energy Cost]
            O3[Stability]
            O4[Resource Use]
            O5[Quality]
        end
    end
    
    subgraph "Evolution"
        V1[Non-dominated Sort]
        V2[Reference Points]
        V3[Selection]
        V4[Crossover]
        V5[Mutation]
    end
    
    subgraph "Output"
        P1[Pareto Front<br/>50-100 solutions]
        P2[Trade-off Analysis]
        P3[Control Recommendations]
    end
    
    I1 --> I2
    I2 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> O1
    E4 --> O2
    E4 --> O3
    E4 --> O4
    E4 --> O5
    
    O1 --> V1
    O2 --> V1
    O3 --> V1
    O4 --> V1
    O5 --> V1
    
    V1 --> V2
    V2 --> V3
    V3 --> V4
    V4 --> V5
    V5 --> E1
    
    V1 --> P1
    P1 --> P2
    P2 --> P3
    
    style E1 fill:#f9b
    style E2 fill:#f9b
    style E3 fill:#f9b
    style E4 fill:#f9b
```

## Performance Metrics Flow

```mermaid
graph LR
    subgraph "Data Collection"
        DC1[Pipeline Metrics]
        DC2[GPU Metrics]
        DC3[Model Metrics]
        DC4[System Metrics]
    end
    
    subgraph "Prometheus"
        P1[Time Series DB]
        P2[Alerting Rules]
        P3[Exporters]
    end
    
    subgraph "Visualization"
        V1[Grafana Dashboards]
        V2[Performance Reports]
        V3[Alert Notifications]
    end
    
    DC1 --> P3
    DC2 --> P3
    DC3 --> P3
    DC4 --> P3
    
    P3 --> P1
    P1 --> P2
    P1 --> V1
    P2 --> V3
    V1 --> V2
```

## Error Handling Flow

```mermaid
flowchart TB
    subgraph "Error Detection"
        ED1{Data Quality?}
        ED2{GPU Available?}
        ED3{Model Loaded?}
        ED4{DB Connected?}
    end
    
    subgraph "Fallback Strategies"
        FS1[Skip Window]
        FS2[Use CPU]
        FS3[Load Backup]
        FS4[Retry Connection]
    end
    
    subgraph "Recovery"
        R1[Log Error]
        R2[Alert Ops]
        R3[Continue Processing]
        R4[Graceful Shutdown]
    end
    
    ED1 -->|Bad| FS1
    ED2 -->|No| FS2
    ED3 -->|No| FS3
    ED4 -->|No| FS4
    
    FS1 --> R1 --> R3
    FS2 --> R1 --> R3
    FS3 --> R1 --> R3
    FS4 --> R1 --> R4
    
    ED1 -->|Good| Normal[Normal Processing]
    ED2 -->|Yes| Normal
    ED3 -->|Yes| Normal
    ED4 -->|Yes| Normal
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development"
        DEV1[Local Docker]
        DEV2[GPU Workstation]
        DEV3[Test Database]
    end
    
    subgraph "CI/CD Pipeline"
        CI1[GitHub Actions]
        CI2[Build Images]
        CI3[Run Tests]
        CI4[Push Registry]
    end
    
    subgraph "Production"
        subgraph "Compute Nodes"
            P1[Node 1<br/>CPU: 32 cores<br/>RAM: 128GB]
            P2[Node 2<br/>GPU: 4x A100<br/>RAM: 256GB]
        end
        
        subgraph "Storage"
            S1[NVMe Array<br/>10TB]
            S2[Object Storage<br/>Models & Results]
        end
        
        subgraph "Network"
            N1[Load Balancer]
            N2[Internal Network<br/>10Gbps]
        end
    end
    
    DEV1 --> CI1
    CI1 --> CI2
    CI2 --> CI3
    CI3 --> CI4
    CI4 --> P1
    CI4 --> P2
    
    N1 --> P1
    N1 --> P2
    P1 -.-> N2
    P2 -.-> N2
    N2 -.-> S1
    N2 -.-> S2
    
    style P2 fill:#f9b,stroke:#333,stroke-width:2px
```

## Data Flow Summary

```mermaid
journey
    title Greenhouse Data Journey
    section Ingestion
      Raw CSV Files: 5: Rust
      Parse & Validate: 4: Rust
      Batch Insert: 5: Rust
    section Processing
      Aggregate Hourly: 4: Rust
      Fill Small Gaps: 3: Rust
      Extract Features: 5: Python, GPU
      Create Eras: 4: Rust
    section Modeling
      Load Features: 5: Python
      Train Models: 4: Python, GPU
      Validate: 4: Python
    section Optimization
      Initialize MOEA: 5: Python
      Evaluate Population: 5: Python, GPU
      Generate Pareto: 5: Python
    section Results
      Control Strategies: 5: User
      Energy Savings: 5: User
      Plant Growth: 5: User
```