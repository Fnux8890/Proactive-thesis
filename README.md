### 🌱 Data-Driven Greenhouse Climate Control & Optimization 🌿
**Optimizing plant growth & energy efficiency through proactive simulation, multi-objective optimization, and investigation of advanced computational acceleration.**

This project develops an advanced **data-driven climate control and optimization system** for greenhouses. It aims to enhance plant health and resource efficiency by integrating **proactive strategies**, **species-specific plant growth simulation** (initially focusing on *Kalanchoe blossfeldiana*), and **multi-objective evolutionary algorithms (MOEAs)**. Building upon concepts from frameworks like DynaGrow, this work seeks to provide more tailored and computationally feasible solutions for sustainable greenhouse management.

### 🔬 **How It Works**
Leveraging historical data, predictive modeling, and (future) IoT-driven environmental adjustments, this system dynamically balances energy efficiency and plant growth. Key aspects include:

- **Data Pipeline & Feature Engineering**: Ingesting historical greenhouse data, performing preprocessing (including era detection based on techniques like PELT/BOCPD/HMM), and extracting features crucial for the subsequent simulation and optimization phases (FR-1, FR-2).
- **Species-Specific Plant Simulation**: Implementing and calibrating plant growth models (e.g., for Kalanchoe) to predict responses to various control strategies (FR-3, NFR-6.1). This is a core component for evaluating potential outcomes.
- **Multi-Objective Optimization (MOEA)**: Employing algorithms like NSGA-II to find Pareto-optimal control strategies that balance conflicting objectives, such as minimizing energy costs and maximizing simulated plant growth (FR-4).
- **Addressing Computational Bottlenecks**: A primary research focus is tackling the significant computational demands of detailed plant-specific simulations and extensive MOEA evaluations. This involves:
  - Identifying performance limitations on traditional CPU architectures.
  - **Investigating and implementing GPU acceleration using CUDA** for critical components (plant model simulation, MOEA fitness evaluations) across various GPU tiers (consumer, professional, data-center class).
- **Scalable, Modular Software Architecture**: Designing the system (primarily Python for simulation/optimization, potentially orchestrated with Elixir/Docker for broader pipeline elements) for extensibility and real-world integration (NFR-2, NFR-3, NFR-4).

### 🚀 **Key Features**
✅ **Species-Specific Plant Simulation**: Utilizing detailed plant models (e.g., for Kalanchoe) to evaluate the outcomes of different control strategies as part of the MOEA fitness evaluation.
✅ **Multi-Objective Optimization**: Finding optimal trade-offs between energy costs and plant productivity using MOEAs.
✅ **Computational Performance Research**: Actively investigating and developing methods to accelerate complex simulations and optimizations using GPU (CUDA) technology.
✅ **Data-Driven Insights**: Utilizing historical data to inform simulations and guide optimization processes.
✅ **Energy-Efficient Growth Strategies**: Aiming to identify control schedules that maximize plant productivity while minimizing resource consumption.
✅ **Modular & Extensible Design**: Facilitating the integration of new plant models, optimization algorithms, or data sources.
✅ **Sustainability-Focused**: Contributing to reducing the carbon footprint and operational costs in horticulture.

### 📚 **Research-Backed & State-of-the-Art**
This project builds upon foundational work in greenhouse control (e.g., DynaGrow) and advances it by:
- Focusing on **species-specific modeling** to provide more accurate and tailored optimization.
- Addressing the **computational feasibility** of advanced modeling and optimization techniques through **GPU acceleration research**.
- Implementing **multi-objective evolutionary algorithms** for sophisticated, data-driven control strategy generation.
- Systematically analyzing **performance trade-offs across different hardware architectures (CPU vs. GPU)** for horticultural optimization tasks.

**Contributing to sustainable agriculture**, this work aligns with Denmark’s **green transition goals**, ensuring **future-proof, resource-conscious food production.**
