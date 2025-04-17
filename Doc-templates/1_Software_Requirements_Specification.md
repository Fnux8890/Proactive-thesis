# Software Requirements Specification (SRS)

**Project Title:** Simulation-Based Greenhouse Climate Control Optimization System
**Document Version:** 1.0
**Last Updated:** April 12, 2025

---

## 1. Introduction

### 1.1 Purpose

This SRS document provides a complete description of the requirements for the Simulation-Based Greenhouse Climate Control Optimization System. The purpose of the system is to explore strategies for balancing energy efficiency and simulated plant growth in greenhouses, building upon concepts from the DynaGrow project. It leverages ingested historical environmental data and uses plant growth simulation coupled with multi-objective optimization algorithms to determine potentially cost-effective control strategies. This document outlines the functional and non-functional requirements for this Master's thesis project, intended for the student developer and supervisor (Jan Corfixen Sørensen).

### 1.2 Scope

The system will focus on demonstrating the feasibility of optimizing simulated greenhouse control based on historical data. The scope includes:

- **Accessing and preparing ingested data:** Utilizing previously loaded historical greenhouse data (temperature, humidity, CO2) stored in a TimescaleDB database. Calculating derived features necessary for simulation and optimization objectives.
- **Simulating plant growth:** Implementing and configuring a plant growth simulation model (e.g., based on photosynthesis principles relevant to DynaGrow) to estimate growth potential based on environmental inputs.
- **Applying multi-objective optimization:** Implementing a Multi-Objective Evolutionary Algorithm (MOEA) to find trade-off solutions between minimizing estimated energy cost and maximizing simulated plant growth.
- **Generating control recommendations:** Outputting the resulting (near-)optimal control strategies (e.g., supplemental light schedules) based on the optimization process.
- **Logging:** Recording key parameters, inputs, outputs, and objective values for analysis and evaluation.

**Out of Scope for initial iterations:** Direct control of physical greenhouse actuators, real-time data ingestion (FR-1.4 is optional), integration with live external APIs like weather/energy (FR-1.5 is optional), development of novel simulation models or optimization algorithms (focus is on applying existing concepts).

### 1.3 Definitions, Acronyms, and Abbreviations

- **SRS:** Software Requirements Specification
- **MOEA:** Multi-Objective Evolutionary Algorithm
- **FR:** Functional Requirement
- **NFR:** Non-Functional Requirement
- **DB:** Database (specifically TimescaleDB in this project)
- **PAR:** Photosynthetically Active Radiation (relevant context from DynaGrow)
- **DLI:** Daily Light Integral (relevant context from DynaGrow)

### 1.4 References

- *Problem statement - masters thesis in Software engineering-6.pdf* (Defines initial objectives)
- *DynGrowManual.pdf* (Provides context on system being enhanced, example models/specs)
- *Dynagrow - multiobjective.pdf* (Details MOEA approach, objectives used in DynaGrow)
- *(Potentially)* *thesis.pdf* (AGFACAND thesis - for implementation concepts like MOEA structure, model handling)

### 1.5 Overview of the Document

This document is structured as follows:

- **Section 2:** Provides an overall description of the system including its context, functions, and operational environment.
- **Section 3:** Details the specific functional and non-functional requirements.
- **Section 4:** Lists any ancillary information such as appendices.

---

## 2. Overall Description

### 2.1 Product Perspective

This system is a software simulation and optimization tool designed as part of a Master's thesis project. It utilizes pre-processed historical data stored in a TimescaleDB database to simulate plant growth under different control strategies and identify optimized strategies using MOEAs. It builds upon concepts demonstrated in the DynaGrow system but focuses on simulation rather than direct hardware control. It serves as a platform for investigating the trade-offs between energy use and plant growth in a simulated environment based on real historical conditions.

### 2.2 Product Functions

The major functionalities provided by the system include:

- **Data Access:** Retrieving relevant historical environmental data from the TimescaleDB.
- **Input Preparation:** Calculating derived features needed for simulation/optimization.
- **Plant Growth Simulation:** Simulating plant responses based on environmental data and configurable models.
- **Multi-Objective Optimization:** Finding near-optimal control strategies balancing energy and growth using MOEAs.
- **Recommendation Output:** Providing the optimized control strategy (e.g., light plan).
- **Logging:** Recording process details for analysis.

### 2.3 User Characteristics

The primary users of this system are:

- **Student Developer (Fnux8890):** Responsible for designing, implementing, testing, and evaluating the system.
- **Supervisor (Jan Corfixen Sørensen):** Responsible for guiding the project, reviewing progress, and evaluating the final thesis based on the system's results.
- *(Secondary)* **Future Researchers:** May use the system or findings as a basis for further work.

### 2.4 Constraints

- **Project Timeline:** Must be completed within the Master's thesis timeframe.
- **Methodology:** Must utilize existing, established simulation modeling and optimization techniques (no novel algorithm invention required).
- **Data Source:** Relies on the specific, pre-ingested historical datasets.
- **Performance:** Simulation and optimization cycles need to complete within a reasonable time for iterative testing and experimentation (Ref NFR-1.1).
- **Modularity/Extensibility:** Design must allow for potential modification or replacement of simulation/optimization components (Ref NFR-3).

### 2.5 Assumptions and Dependencies

- The pre-ingested historical data in TimescaleDB is sufficient and representative for the project's goals.
- Necessary software libraries for database access, simulation implementation (if needed), and MOEA (if using libraries) are available.
- The chosen plant growth simulation model, although potentially simplified, provides a qualitatively reasonable representation for the optimization task (Ref NFR-6.1).
- Sufficient computational resources are available on the development machine for running simulations and optimizations.
- Access to relevant literature for selecting appropriate simulation model parameters and MOEA configurations.

---

## 3. Specific Requirements

### 3.1 Functional Requirements

#### FR-1: Data Ingestion and Management (Status: Completed)

- **FR-1.1:** The system needs to extract historical greenhouse environmental data which includes temperature and humidity along with CO2 and Light intensity from predefined file formats like CSV or JSON. *(Implied: Already done)*
- **FR-1.2:** The system will parse incoming data into suitable data types to facilitate easier processing in future stages. *(Implied: Already done)*
- **FR-1.3:** The Ingestion system must save the data into a database that enables efficient querying. *(Implied: Already done using TimescaleDB)*
- **FR-1.4 (Optional):** Future system architecture extensions must enable real-time sensor data stream ingestion.
- **FR-1.5 (Optional):** System architecture needs to provide functionality for integrating external data sources like weather forecasts.

#### FR-2: Input data preparation & Feature definition

- **FR-2.1:** The system shall retrieve historical database information from TimescaleDB that matches the simulation and optimization time windows.
- **FR-2.2:** The system shall calculate or allow definition of important input features required for simulation models or optimization objective functions (e.g., PAR sums, energy cost estimates).

#### FR-3: Plant Growth simulation

- **FR-3.1:** The system shall implement an appropriate plant growth simulation model (e.g., based on photosynthesis principles).
- **FR-3.2:** The system shall accept configurable parameters specific to the chosen plant growth model.
- **FR-3.3:** The simulation model shall process specified inputs (from FR-2.1/FR-2.2) and produce relevant metrics (e.g., growth index).

#### FR-4: Multi-Objective Optimization

- **FR-4.1:** The system shall implement a MOEA (Multi-Objective Evolutionary Algorithm) for finding optimal control strategies.
- **FR-4.2:** The MOEA shall assess possible control strategies by utilizing the plant growth simulation (FR-3) and the established objective functions.
- **FR-4.3:** The system shall implement at least two objective functions: (a) minimization of estimated energy cost, and (b) maximization of the simulated plant growth metric.
- **FR-4.4:** The system shall allow key MOEA parameters (e.g., population size, generations) to be configurable.

#### FR-5: Control Recommendation and Logging

- **FR-5.1:** The system shall output the determined (near-)optimal control strategy (e.g., light plan, setpoints) resulting from the MOEA.
- **FR-5.2:** The system shall log key decision parameters, inputs, objective function values, and the recommended control strategy for each optimization run.

### 3.2 Non-Functional Requirements

#### NFR-1: Performance (Operational Concern)

- **NFR-1.1:** Each full simulation and optimization cycle (data reading, simulation, MOEA, recommendation generation) shall finish in under 5 minutes to facilitate reasonable experimentation time.

#### NFR-2: Scalability (Architectural Concern)

- **NFR-2.1:** The system design should permit increasing optimization problem complexity (e.g., longer planning horizons, more objectives) without requiring fundamental architectural changes, though performance impacts are expected.

#### NFR-3: Extensibility (Life-cycle Concern)

- **NFR-3.1:** The architectural design shall support the substitution of the plant growth simulation model (FR-3.1) by alternative models through a standardized interface.
- **NFR-3.2:** The system architecture shall enable swapping of MOEA implementations (FR-4.1) and modification of objective functions (FR-4.3) while minimizing the effect on other modules.

#### NFR-4: Maintainability & Modularity (Life-cycle Concern)

- **NFR-4.1:** The codebase structure shall follow a modular design which divides responsibilities among data access, simulation logic, optimization algorithms, objective functions, and logging systems.
- **NFR-4.2:** The documentation for code shall enable clear understanding of both module structures and their interactions.

#### NFR-5: Reliability (Operational Concern)

- **NFR-5.1:** The system shall handle potential errors during database access operations gracefully (e.g., manage connection problems, missing data).
- **NFR-5.2:** The simulation model component shall manage invalid input parameters or states appropriately (e.g., log errors, return defined error values).
- **NFR-5.3:** The optimization component shall address potential convergence problems or numerical instability gracefully (e.g., return best-found solution after timeout, log warnings).

#### NFR-6: Model Validity (Operational/Research Concern)

- **NFR-6.1:** The implemented plant growth simulation model requires validation (method TBD) to confirm that its outputs align, at least qualitatively, with known horticultural principles or literature findings.

### 3.3 Interface Requirements

#### User Interfaces

- **Primary Interface:** Command-line execution for running simulations/optimizations.
- **Configuration:** Input parameters for simulation models and MOEA likely managed via configuration files (e.g., JSON, YAML).
- **Output:** Results (optimized plans, logs) written to console or output files. Graphical User Interface (GUI) is out of scope.

#### Hardware Interfaces

- None directly. The system interacts with data already stored in the database, not live sensors or actuators.

#### Software Interfaces

- **Database Interface:** Interaction with TimescaleDB via appropriate database drivers/libraries (e.g., JDBC if applicable).
- **Internal Module Interfaces:** Well-defined programmatic interfaces between the Data Access, Input Preparation, Simulation, MOEA, Objective Functions, and Logging modules (Ref NFR-3.1, NFR-3.2, NFR-4.1).
- **External Interfaces (Optional):** If FR-1.5 is implemented, requires interfaces to specified weather/energy APIs (e.g., RESTful APIs).

### 3.4 Data Requirements

- **Input Data:** Access to the historical time-series data (Timestamp, Temp, Humidity, CO2, etc.) stored in TimescaleDB. Format as defined during ingestion (FR-1).
- **Configuration Data:** Parameters for the selected plant simulation model; parameters for the MOEA. Format TBD (likely config files).
- **Output Data:** Generated control strategies (e.g., light plans as time-indexed schedules). Log files containing execution details, parameters, objective values. Format TBD.
- **Data Retention:** Log files and key results should be retained for thesis analysis and evaluation. Long-term archival policies are not a primary concern for this project phase.

### 3.5 System Features Summary

| Feature                           | Description                                                                     |
| :-------------------------------- | :------------------------------------------------------------------------------ |
| Data Access & Prep              | Retrieves historical data, prepares inputs/features for simulation/optimization |
| Plant Growth Simulation         | Simulates plant growth response based on environmental inputs & model params      |
| Multi-Objective Optimization    | Finds trade-off control strategies using MOEA (balancing energy/growth)         |
| Control Recommendation & Output | Generates the optimized control plan determined by the MOEA                     |
| Logging                           | Records key information about the simulation and optimization process           |

---

## 4. Appendices

### Appendix A: Glossary

- **MOEA:** Multi-Objective Evolutionary Algorithm - Optimization technique used to find trade-offs between conflicting goals.
- **TimescaleDB:** Time-series database used for storing historical sensor data.
- **Simulation Model:** A mathematical representation used to predict plant growth responses.

### Appendix B: Assumptions and Dependencies Documentation

- Assumes validity and sufficiency of ingested historical data.
- Assumes availability of standard libraries for DB access, potential MOEA implementation.
- Depends on the qualitative validity of the chosen simulation model.

### Appendix C: Revision History

| Version | Date           | Description                        | Author      |
| :------ | :------------- | :--------------------------------- | :---------- |
| 1.0     | April 12, 2025 | Initial version based on project scope | Fnux8890/AI |

---

**Approval and Sign-Off**

*(Placeholders for supervisor sign-off)*

| Name                  | Title      | Signature | Date     |
| :-------------------- | :--------- | :-------- | :------- |
| Jan Corfixen Sørensen | Supervisor |           |          |
|                       |            |           |          |
