# Pipeline Flow Diagrams

## Traditional Pipeline vs Sparse Pipeline

### Traditional Pipeline Flow (Failed Approach)

```mermaid
flowchart TB
    subgraph "Traditional Multi-Container Pipeline"
        A[Raw CSV/JSON Files] --> B[Rust Ingestion Container]
        B --> C[(PostgreSQL)]
        C --> D[Python Preprocessing Container]
        
        D --> E{Time Regularization}
        E -->|5-min intervals| F[91.3% NULL values!]
        F --> G{Imputation}
        G -->|Forward Fill| H[Stale Data]
        G -->|Interpolation| I[Synthetic Data]
        
        H --> J[Rust Era Detection Container]
        I --> J
        
        J --> K{PELT Algorithm}
        K -->|Every gap is changepoint| L[850,000 False Eras]
        
        L --> M[Python Feature Extraction]
        M -->|Out of Memory| N[FAILURE]
        
        style N fill:#f96,stroke:#333,stroke-width:4px
        style F fill:#ff9,stroke:#333,stroke-width:2px
        style L fill:#ff9,stroke:#333,stroke-width:2px
    end
```

### Sparse Pipeline Flow (Working Solution)

```mermaid
flowchart TB
    subgraph "Sparse Single-Container Pipeline"
        A[Raw CSV/JSON Files] --> B[Rust Ingestion]
        B --> C[(PostgreSQL)]
        
        C --> D[GPU Sparse Pipeline Container]
        
        subgraph D["Integrated Sparse Processing"]
            D1[Stage 1: Hourly Aggregation]
            D2[Stage 2: Conservative Gap Fill]
            D3[Stage 3: GPU Feature Extraction]
            D4[Stage 4: Era Creation]
            
            D1 -->|Coverage Metrics| D2
            D2 -->|Max 2hr gaps| D3
            D3 -->|Adaptive Windows| D4
        end
        
        D --> E[Sparse Features & Eras]
        E --> F[Model Builder]
        F --> G[MOEA Optimizer]
        
        style D fill:#9f9,stroke:#333,stroke-width:2px
        style G fill:#9f9,stroke:#333,stroke-width:4px
    end
```

## Detailed Stage Flow

### Stage 1: Intelligent Aggregation

```mermaid
flowchart LR
    subgraph "Input Data Pattern"
        A1[00:00 - Temp: 22.5]
        A2[00:05 - NULL]
        A3[00:10 - NULL]
        A4[00:15 - CO2: 450]
        A5[00:20 - NULL]
        A6[00:25 - Humidity: 65%]
    end
    
    subgraph "Hourly Aggregation"
        B[SQL GROUP BY hour]
        B --> C{Coverage >= 10%?}
        C -->|Yes| D[Keep Hour]
        C -->|No| E[Discard Hour]
    end
    
    subgraph "Output"
        F[Hour: 00:00<br/>Temp: 22.5 (1 sample)<br/>CO2: 450 (1 sample)<br/>Humidity: 65% (1 sample)<br/>Coverage: OK]
    end
    
    A1 --> B
    A4 --> B
    A6 --> B
    D --> F
```

### Stage 2: Conservative Gap Filling

```mermaid
flowchart TB
    subgraph "Before Gap Filling"
        A[Hour 1: 22.5°C]
        B[Hour 2: NULL]
        C[Hour 3: NULL]
        D[Hour 4: 23.1°C]
        E[Hour 5: NULL]
        F[Hour 6: NULL]
        G[Hour 7: NULL]
        H[Hour 8: 24.2°C]
    end
    
    subgraph "Gap Fill Logic"
        I{Gap <= 2 hours?}
        I -->|Yes| J[Forward Fill]
        I -->|No| K[Keep NULL]
    end
    
    subgraph "After Gap Filling"
        L[Hour 1: 22.5°C]
        M[Hour 2: 22.5°C ←filled]
        N[Hour 3: 22.5°C ←filled]
        O[Hour 4: 23.1°C]
        P[Hour 5: 23.1°C ←filled]
        Q[Hour 6: 23.1°C ←filled]
        R[Hour 7: NULL ←gap too large]
        S[Hour 8: 24.2°C]
    end
    
    A --> I
    B --> I
    C --> I
    I --> L
    I --> M
    I --> N
    I --> O
    I --> P
    I --> Q
    I --> R
    I --> S
```

### Stage 3: Adaptive Windowing

```mermaid
flowchart TB
    subgraph "Data Quality Analysis"
        A[Calculate Metrics]
        A --> B[Coverage: 73%]
        A --> C[Continuity: 0.85]
        A --> D[Consistency: 0.62]
        A --> E[Overall Score: 0.73]
    end
    
    subgraph "Window Configuration"
        E --> F{Quality Score}
        F -->|>= 0.8| G[24-hour windows<br/>25% overlap]
        F -->|0.6-0.8| H[12-hour windows<br/>50% overlap]
        F -->|0.4-0.6| I[6-hour windows<br/>75% overlap]
        F -->|< 0.4| J[3-hour windows<br/>75% overlap]
    end
    
    subgraph "Window Creation"
        H --> K[Window 1: 00:00-12:00]
        H --> L[Window 2: 06:00-18:00]
        H --> M[Window 3: 12:00-24:00]
        H --> N[Window 4: 18:00-06:00]
    end
    
    style H fill:#ff9,stroke:#333,stroke-width:2px
```

### Stage 4: Era Creation Flow

```mermaid
flowchart LR
    subgraph "Feature Windows"
        A[Jan Features<br/>16 windows]
        B[Feb Features<br/>28 windows]
        C[Mar Features<br/>31 windows]
    end
    
    subgraph "Monthly Grouping"
        D[Group by Month]
        E[Calculate Statistics]
    end
    
    subgraph "Era Output"
        F[Era 1: January<br/>Avg Temp: 8.4°C<br/>Features: 16]
        G[Era 2: February<br/>Avg Temp: 12.3°C<br/>Features: 28]
        H[Era 3: March<br/>Avg Temp: 16.7°C<br/>Features: 31]
    end
    
    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    E --> G
    E --> H
```

## GPU Acceleration Flow

```mermaid
flowchart TB
    subgraph "CPU Preparation"
        A[Adaptive Windows] --> B[Batch Formation]
        B --> C[Pin Memory]
    end
    
    subgraph "GPU Processing"
        D[Transfer to GPU]
        E[Launch Kernels]
        
        subgraph "Parallel Computation"
            F1[Statistical Features]
            F2[Temporal Features]
            F3[Domain Features]
        end
        
        G[Synchronize]
        H[Transfer Results]
    end
    
    subgraph "CPU Finalization"
        I[Unpin Memory]
        J[Format Output]
    end
    
    C --> D
    D --> E
    E --> F1
    E --> F2
    E --> F3
    F1 --> G
    F2 --> G
    F3 --> G
    G --> H
    H --> I
    I --> J
    
    style E fill:#9cf,stroke:#333,stroke-width:2px
    style F1 fill:#9cf,stroke:#333,stroke-width:1px
    style F2 fill:#9cf,stroke:#333,stroke-width:1px
    style F3 fill:#9cf,stroke:#333,stroke-width:1px
```

## Complete System Flow

```mermaid
flowchart TB
    subgraph "Data Sources"
        A1[Greenhouse Sensors]
        A2[Weather Data]
        A3[Energy Prices]
    end
    
    subgraph "Data Ingestion"
        B[Rust Pipeline]
        B --> C[(TimescaleDB)]
    end
    
    subgraph "Sparse Pipeline"
        D[Hourly Aggregation]
        E[Gap Filling]
        F[GPU Features]
        G[Era Creation]
        
        D --> E
        E --> F
        F --> G
    end
    
    subgraph "Model Training"
        H[Load Features]
        I[Train LSTM]
        J[Train LightGBM]
        K[Validate Models]
        
        H --> I
        H --> J
        I --> K
        J --> K
    end
    
    subgraph "MOEA Optimization"
        L[Initialize Population]
        M[GPU Evaluation]
        N[Selection]
        O[Crossover/Mutation]
        P{Converged?}
        Q[Pareto Front]
        
        L --> M
        M --> N
        N --> O
        O --> M
        M --> P
        P -->|No| N
        P -->|Yes| Q
    end
    
    A1 --> B
    A2 --> B
    A3 --> B
    C --> D
    G --> H
    K --> L
    
    style C fill:#99f,stroke:#333,stroke-width:2px
    style F fill:#9cf,stroke:#333,stroke-width:2px
    style M fill:#9cf,stroke:#333,stroke-width:2px
    style Q fill:#9f9,stroke:#333,stroke-width:4px
```

## Data Sparsity Visualization

```mermaid
gantt
    title Data Availability Timeline (January 2014)
    dateFormat HH:mm
    axisFormat %H:%M
    
    section Temperature
    Available :done, temp1, 00:00, 2h
    Missing   :crit, temp2, 02:00, 4h
    Available :done, temp3, 06:00, 3h
    Missing   :crit, temp4, 09:00, 6h
    Available :done, temp5, 15:00, 1h
    Missing   :crit, temp6, 16:00, 8h
    
    section CO2
    Missing   :crit, co21, 00:00, 3h
    Available :done, co22, 03:00, 1h
    Missing   :crit, co23, 04:00, 5h
    Available :done, co24, 09:00, 2h
    Missing   :crit, co25, 11:00, 13h
    
    section Humidity
    Available :done, hum1, 00:00, 1h
    Missing   :crit, hum2, 01:00, 7h
    Available :done, hum3, 08:00, 4h
    Missing   :crit, hum4, 12:00, 12h
```

This visualization shows why traditional approaches fail - the data has more gaps than values!