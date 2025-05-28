# Technical Decision Log (TDL)

## Document Information

- **Title:** Technical Decision Log
- **Project:** Simulation-Based Greenhouse Control Optimization
- **Version:** 1.0 (Draft based on Facilitation Report)
- **Last Updated:** [Date]

---

## 1. Introduction

### 1.1 Purpose

This document logs the significant technical and architectural decisions made throughout the project lifecycle. It ensures that each decision is well-documented, justified, and traceable for future reference, particularly guiding the implementation within the Master's thesis scope.

### 1.2 Decision Template

Every decision record follows the structure outlined below:

- **ID:** Unique identifier for the decision.
- **Date:** Date the decision was effectively made or finalized.
- **Context:** Background information and the problem being addressed, referencing project goals or constraints.
- **Decision:** The resolution or approach selected.
- **Alternatives Considered:** Other options evaluated before choosing the final solution (if applicable).
- **Consequences:** Impacts, benefits, and trade-offs associated with the chosen approach, including implications for the project timeline or scope.
- **Status:** The current state of the decision (e.g., Approved, Under Review).

---

## 2. Architectural Decisions

### 2.1 Core System Architecture

- **ID:** AD-001
- **Date:** [Date based on Facilitation Report/Project Start]
- **Context:**
  - Need for a structured approach to handle data flow from historical records through simulation and optimization to results.
  - Requirement for testability and focused development within an iterative methodology.
  - Alignment with Master's thesis scope focusing on simulation and MOEA integration.
- **Decision:**
  - Implement a modular architecture comprising distinct components: Data Access, Input Preparation, Plant Simulation, Multi-Objective Optimization (MOEA), and Logging/Output.
- **Alternatives Considered:**
  - Monolithic design.
  - Microservices architecture (overkill for project scope).
- **Consequences:**
  - **Pros:** Facilitates iterative development, allows parallel work (component implementation & report writing), improves testability, clearer separation of concerns.
  - **Cons:** Requires careful definition and management of interfaces between modules early on (Ref: Issue #12).
- **Status:** Approved

### 2.2 Development Methodology

- **ID:** AD-002
- **Date:** [Date based on Facilitation Report]
- **Context:**
  - Limited timeframe (~2 months remaining) for development and report writing.
  - Need to demonstrate a functional end-to-end system addressing core objectives.
  - Requirement for parallel report writing alongside implementation.
- **Decision:**
  - Adopt an Iterative Development methodology with short cycles (1-2 weeks), focusing on building a minimum viable end-to-end workflow first, followed by refinement. Prioritize tasks based on GitHub Issues (#8-12).
- **Alternatives Considered:**
  - Waterfall model (unsuitable for time constraints and potential unknowns).
  - Agile/Scrum (formal Scrum may be too heavyweight for a single-person thesis project).
- **Consequences:**
  - **Pros:** Allows for early demonstration of core functionality, provides flexibility to adapt, facilitates parallel writing by documenting completed iterations/components.
  - **Cons:** Requires strict adherence to iteration goals and timelines; risk of integration issues if interfaces are not managed well.
- **Status:** Approved

---

## 3. Technology Choices

### 3.1 Primary Programming Language

- **ID:** TC-001
- **Date:** [Date based on Project Start/Iteration 1]
- **Context:**
  - Need for a performant language suitable for simulation and potentially computationally intensive optimization tasks.
  - Desire for memory safety and strong typing.
  - Previous project experience or requirement.
- **Decision:**
  - Utilize Rust as the primary language for the core application logic (Data Access, Input Prep, Simulation, MOEA, Logging).
- **Alternatives Considered:**
  - Python (Common in data science/simulation, but potentially slower execution speed for core loops; integration with Rust could be an option but adds complexity).
  - C++ (Performant, but potentially steeper learning curve or less memory safety focus than Rust).
- **Consequences:**
  - **Pros:** High performance, memory safety, strong type system, growing ecosystem for scientific computing.
  - **Cons:** Potentially longer development time compared to Python for certain tasks; requires careful management of dependencies (crates).
- **Status:** Approved

### 3.2 Data Storage for Historical Data

- **ID:** TC-002
- **Date:** [Date based on Project Start/Iteration 1]
- **Context:**
  - Requirement to store and efficiently query historical time-series environmental data (temperature, humidity, etc.).
  - Need to integrate with the Rust application for data retrieval.
- **Decision:**
  - Use TimescaleDB (PostgreSQL extension) as the database for storing historical input data.
- **Alternatives Considered:**
  - InfluxDB (Another popular time-series DB).
  - Standard PostgreSQL (Less optimized for time-series queries).
  - File-based storage (CSV, Parquet - less efficient for querying specific time ranges or variables).
- **Consequences:**
  - **Pros:** Optimized for time-series data, leverages robust PostgreSQL foundation, good integration with Rust (`sqlx` crate).
  - **Cons:** Adds a dependency on a specific database system; requires understanding of TimescaleDB specific features (hypertables, etc.).
- **Status:** Approved

### 3.3 Deployment and Environment Consistency

- **ID:** TC-003
- **Date:** [Date based on Project Start/Iteration 1]
- **Context:**
  - Need for a consistent runtime environment across development and potential testing/execution phases.
  - Simplification of dependency management for the Rust application and database.
- **Decision:**
  - Utilize Docker containers for packaging and running the Rust application and potentially the TimescaleDB instance during development/testing.
- **Alternatives Considered:**
  - Manual installation of dependencies on host machines (prone to inconsistencies).
  - Virtual Machines (heavier than containers).
- **Consequences:**
  - **Pros:** Ensures consistent environment, simplifies dependency management, facilitates reproducibility.
  - **Cons:** Requires familiarity with Docker concepts and Dockerfile creation.
- **Status:** Approved

### 3.4 Multi-Objective Evolutionary Algorithm (MOEA) Approach

- **ID:** TC-004
- **Date:** [Date based on Facilitation Report]
- **Context:**
  - Requirement to implement Objective 2: Use an MOEA to balance energy cost and simulated growth.
  - Constraint to use established algorithms due to time limits (avoid novel algorithm development).
- **Decision:**
  - Implement a standard, established MOEA such as NSGA-II or SPEA2. Prioritize using an existing Rust library (e.g., `argmin`, `metaheuristics-rs`, or others if suitable) over a full custom implementation if possible and time-efficient.
- **Alternatives Considered:**
  - Single-objective optimization (does not meet the core project aim).
  - Developing a novel MOEA (out of scope and too time-consuming).
  - Heuristic or rule-based optimization (less likely to find non-dominated trade-offs).
- **Consequences:**
  - **Pros:** Leverages well-understood algorithms, reduces development time if a suitable library is found, focuses effort on integration rather than algorithm design.
  - **Cons:** Selected library might have limitations or require adaptation; performance tuning of MOEA parameters might still be needed.
- **Status:** Approved

### 3.5 Plant Simulation Model Approach

- **ID:** TC-005
- **Date:** [Date based on Facilitation Report]
- **Context:**
  - Requirement to implement Objective 1: Simulate plant growth based on environmental inputs.
  - Constraint to use established concepts and avoid novel model development.
  - Acknowledged limitation (NFR-6) that quantitative validation against real plant data is out of scope.
- **Decision:**
  - Develop or adapt a simulation model based on established concepts, such as photosynthesis models referenced in related work (e.g., DynaGrow documentation). Focus on achieving qualitatively correct behavior based on horticultural principles.
- **Alternatives Considered:**
  - Using complex, pre-existing plant simulation software (potentially difficult to integrate or customize for MOEA interaction).
  - Developing a highly detailed, physiologically accurate model (out of scope).
  - Using a purely statistical model (might not capture dynamic responses needed for control strategy evaluation).
- **Consequences:**
  - **Pros:** Feasible within the project scope, leverages existing knowledge, allows focus on integration with MOEA.
  - **Cons:** Simulation results are primarily qualitative; model limitations must be clearly documented; requires careful selection/adaptation of model equations.
- **Status:** Approved

### 3.6 Primary Implementation Language Strategy

- **ID:** TC-006
- **Date:** [Date - Reflecting current decision]
- **Context:**
  - Need to balance development speed with potential performance requirements within the tight Master's thesis schedule.
  - Mature libraries exist in Python for data science and MOEA tasks.
  - Rust offers potential performance benefits, especially for computationally intensive simulation.
- **Decision:**
  - Adopt a **Python-first strategy** for core components: Data Access (`psycopg2`/`sqlalchemy`), Input Preparation (`pandas`/`polars`), MOEA (`pymoo`), and the initial implementation of the Plant Simulation model.
  - Reserve Rust as a **potential performance optimization** specifically for the core Simulation logic, to be implemented only if profiling of the Python version indicates it's necessary to meet NFR-1.1. Integration would occur via Python FFI (e.g., `pyo3`).
- **Alternatives Considered:**
  - Rust-only implementation (Potentially slower development, less mature libraries for MOEA/data science).
  - Python-only implementation (Risk of simulation performance bottleneck).
- **Consequences:**
  - **Pros:** Maximizes development speed using mature libraries, addresses key functional requirements quickly, provides a clear path for performance optimization if needed without over-engineering upfront.
  - **Cons:** Introduces potential complexity if FFI is required later; requires careful profiling to determine if the Rust simulation component is needed.
- **Status:** Approved

---

## 4. Implementation Decisions

*(Implementation decisions will be logged here as they are made during development iterations, e.g., choice of specific Python/Rust crates, simulation parameter values, MOEA configuration, data handling details)*

### 4.1 Choice of MOEA Library

- **ID:** ID-001
- **Date:** [Date - Reflecting current decision]
- **Context:** Need to select a Python library for MOEA implementation (Decision TC-004 & TC-006). Evaluation criteria: ease of use, flexibility for custom objectives, documentation, maintenance status.
- **Decision:** Selected [Library Name, e.g., `pymoo`].
- **Alternatives Considered:** [Other libraries evaluated, e.g., `DEAP`, `platypus-opt`], Custom implementation.
- **Consequences:** [Pros/Cons of the chosen library based on evaluation].
- **Status:** Proposed / Approved *(Adjust as applicable)*

### 4.2 Choice of Data Access Library (Python)

- **ID:** ID-002
- **Date:** [Date - Reflecting current decision]
- **Context:** Need to select Python libraries for accessing TimescaleDB (Decision TC-006). Evaluation criteria: compatibility with TimescaleDB/PostgreSQL, ease of use, integration with DataFrame libraries.
- **Decision:** Selected [Library Names, e.g., `psycopg2` and potentially `SQLAlchemy`].
- **Alternatives Considered:** Other database drivers.
- **Consequences:** [Pros/Cons of the chosen libraries based on evaluation].
- **Status:** Proposed / Approved *(Adjust as applicable)*

### 4.3 Choice of Data Preparation Library (Python)

- **ID:** ID-003
- **Date:** [Date - Reflecting current decision]
- **Context:** Need to select Python library for data manipulation and feature extraction (Decision TC-006). Evaluation criteria: performance, API richness for time-series, ease of use.
- **Decision:** Selected [Library Name, e.g., `polars` or `pandas`]. *(Polars often preferred for performance)*
- **Alternatives Considered:** `pandas` (if `polars` chosen), `numpy`/`scipy` directly (less convenient).
- **Consequences:** [Pros/Cons of the chosen library based on evaluation].
- **Status:** Proposed / Approved *(Adjust as applicable)*

*(Add more implementation decisions as they are made)*

---

## 5. Decision Review Process

Decisions are primarily guided by the Project Facilitation Report, the refined objectives, and the Master's thesis scope. Significant deviations or new major decisions impacting scope or timeline should be discussed with the supervisor.

---

## 6. Decision Status Tracking

- **Proposed:** Decision under consideration.
- **Approved:** Decision agreed upon and guiding implementation.
- **Superseded:** Decision replaced by a newer one.

*(Note: No code examples are included as decisions are tracked textually in this document.)*

---

## Appendices

### Appendix A: Reference Documents

- Project Facilitation Report
- Refined Requirements Document (v1.0)
- Initial Problem Statement
- GitHub Issues (#1-12)
- DynaGrow Documentation (DynGrowManual.pdf)
