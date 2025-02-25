# Literature Study

Proactive Climate Simulation for Data-Driven Greenhouse Optimization
Author: Michael H. Pedersen Date: 2025-02-07

Preface
This Review literature of topics of how plants operate and software approaches to optimize plant
growth and use different strategies to broaden the climate computers capabilities using data. The
review looks into these two topics as an understand about plants lay the foundation for the domain and
the software strategies and approaches is to find out what has been done already. The work would lay
the foundational knowledge to use for trying to build upon existing approaches and maybe figure out if
there is room for trying to add proactive features to this field.

The review incorporates 22 studies that together should give fundamentals to build upon. Aspects such
as digital twins, building interfaces for climate computers and techniques for pro-activeness are some
core aspects of the papers found within.

This manuscript is written for researchers, students and practitioners advancing climate-resilient
agriculture to those who are interested in the state-of-the-art IoT-ML applications of controlled
environment optimizations.

Introduction
Environmental issues and increasing population have especially highlighted the importance of
sustainable and energy efficient farming practices. Greenhouses are in high demand because they
provide a protected environment which leads to production throughout the year. Nonetheless, the
management of these environments remains a complex process of achieving the optimal plant health
while ensuring optimal energy expenditure for climate control. There is a need for highly developed
techniques that will involve data-based optimization to change the conditions of greenhouses with the
minimum energy consumption and other operational costs.

The application of data-driven techniques for the enhanced climate control of greenhouses has been
discussed in this report. This includes a discussion of post-active and pre-active strategies for climate
control. These advanced strategies incorporated real-time environmental data, local microclimate data,
and predictive modelling to develop a climate control system that could, in fact, anticipate changes and
proactively adjust to them. This is a development of the basic systems studied earlier, such as
DynaGrow [1] and IntelliGrow [2], which both implemented automation and response control with IoT
and machine learning techniques.

The report is divided into two parts: first, a critical review of the literature related to the state of data-
driven climate control technologies in greenhouses through an analysis of various predictive models
and control systems, and second, a consolidation of those divergent methods and operational
frameworks across more than 85 studies on the effectiveness of post-active and pre-active approaches
in plant growth optimization and energy consumption. It also examines multi-objective optimization
and genetic algorithms for integrating historical and real-time data for predictive control.

This thesis part two is going to develop and test a simulated proactive/pre-active climate control using
orchestration based on Docker. This simulation will not only help in implementing the strategies in
different greenhouse conditions but also in the refinement and validation of the strategies identified in
the literature review. It shall present a flexible, low-cost framework that can help in guiding
researchers, students, and practitioners in the right direction of deploying a suitable climate control
solution in controlled environments.

1

The work presented in the following report is a systematic approach towards the understanding of the
conflicts that occur between energy efficiency and optimum plant health in a greenhouse system. This
thesis aims at improving the current knowledge on the application of IoT and machine learning for
climate control to develop better, versatile solutions for sustainable and climate resilient agriculture.

State-of-the-Art

Historical Contributions
• Contribution 1
• Contribution 2

Current Best Solutions
• Solution 1
• Solution 2

Literature Reviews

Paper 1: DynaGrow [1]

Title:

Authors:

DynaGrow – Multi-Objective Optimization for Energy Cost-efficient
Control of Supplemental Light in Greenhouses

• Jan Corfixen Sørensen
• Katrine Heinsvig Kjaer
• Carl-Otto Ottosen
• Bo Nørregaard Jørgensen

Motivation for
doing the
research:

Supplement lighting accounts for 75% of energy consumption in Danish
horticulture in 2009. With the recent surge in energy prices, there is a need to
optimize for fluctuating energy prices, as static lighting strategies lead to
higher costs when prices spike.

The research
problem:

The horticulture industry competes globally, so reducing operational costs is
crucial. Moreover, renewable energy introduction can cause price volatility.
This paper tackles such multi-objective optimization with a MOEA-based,
modular plugin system, leveraging local weather, plant data, and energy price
inputs for continuous, data-driven objective adjustments—hence the name
DynaGrow.

The authors point out that standard greenhouse climate controls do not
adequately address rising energy costs and fluctuating electricity prices while
maintaining plant quality. DynaGrow frames the challenge as a multi-objective
optimization problem, employing an application-specific MOEA to balance
cost savings with plant health in real time.

DynaGrow’s novelty lies in its modular, feature-oriented design that integrates
seamlessly with existing greenhouse equipment. By incorporating local
climate data, weather predictions, and electricity prices, the system can deliver
targeted, adaptive control strategies. Experiments demonstrate substantial cost
and energy savings, making DynaGrow both scalable and practical for
sustainable greenhouse management.

2

Chosen
research
approach /
methodology:

Reported
results:

DynaGrow recasts greenhouse climate control as a multi-objective
optimization problem, applying a specialized genetic algorithm called
CONTROLEUM-GA to optimize trade-offs between cost efficiency and plant
growth quality. It integrates real-time data, including local climate conditions,
weather forecasts, and fluctuating electricity prices, enabling a dynamic, real-
time control system without the need for expensive hardware upgrades.

Their study features experiments in three separate greenhouse compartments: a
standard fixed-rate setup, and two DynaGrow configurations (SON-T and LED
lighting). Key performance metrics include energy use, cost reduction, and
plant quality.

In experiments, DynaGrow reduced energy usage by up to 56% and slashed
costs by up to 64%, all while dynamically adapting to real-time electricity
price changes. Crucially, these energy savings and cost reductions did not
degrade plant quality.

The system’s data-driven optimization and real-world validation confirm that
domain-specific algorithms can effectively constrain resource consumption
while maintaining productivity. This adaptability—and the validation in
operational settings—positions DynaGrow as a forward-looking solution that
marries sustainability with cost-effectiveness.

What are the
contributions /
conclusions?

• Multi-Objective Optimization: Leverages Pareto-based approaches to

simultaneously minimize energy costs and maintain plant health.

• Modular System Architecture: Easily integrates into existing greenhouse

Your
evaluation of
the reported
results with
respect to the
research
objective

infrastructure, promoting broad adoption potential.

• Real-World Validation: Demonstrates significant cost and energy benefits

in actual greenhouse trials.

• Adaptive Control: Incorporates weather forecasts and electricity prices to

achieve continuous cost-efficiency and stable growing conditions.

• Future Extensions: Offers a framework for expanding to other climate

parameters (e.g., temperature and CO₂ management).

• Significance: The findings demonstrate notable energy and cost savings—
two pressing concerns for greenhouse operations—showing that MOEA-
driven solutions are both feasible and beneficial.

• Validity: The inclusion of real-world experiments bolsters the credibility of
the approach. The metrics used (energy consumption, cost, plant quality)
effectively measure DynaGrow’s impact.

• Novelty: Integrating multi-objective optimization with evolving energy

prices and weather forecasts sets DynaGrow apart from standard or single-
objective solutions.

• Feasibility: The system’s modular design is technically viable, requiring

minimal retrofitting to existing climate computers. This facilitates adoption
in the industry.

• Possible Future Work: Broader and longer-term studies (e.g., multiple crop

types, diverse climate conditions) would confirm its scalability and
robustness.

3

Paper 2: AFDACAND [3]

Title of the
paper:

Name(s) of
author /
authors:

Motivation for
doing the
research:

The research
problem /
objective:

Is this
research
original?

Chosen
research
approach /
methodology:

“A Generic Framework for Automatic Configuration of Artificial Neural
Networks for Data Modeling.”

Morten Gill Wollsen, The Maersk Mc-Kinney Moller Institute, The Technical
Faculty, University of Southern Denmark

The thesis addresses the growing availability of sensor data in many domains
(IoT, Big Data, etc.) and highlights the challenge that most existing ANN tools
require deep expertise in network architecture, configuration, and
hyperparameter tuning. AGFACAND (the proposed framework) aims to
provide a more user-friendly, automated way to create ANN-based models,
thus reducing the barrier for non-experts.

1. Automate the selection and configuration of different types of ANNs (e.g.,

MLP, ELM, SVR) for various data modeling tasks.

2. Reduce the amount of expert involvement needed when using ANNs.
3. Demonstrate that an automated framework can yield competitive or
superior results compared to manual, expert-driven approaches.

Yes. While many software libraries exist for training neural networks,
AGFACAND offers a “one-click” style of automation that combines feature
selection, network type selection, hyperparameter tuning (via Bayesian
Optimization), and evaluation. This end-to-end automation is what makes the
contribution unique.

1. Feature Selection: Uses algorithms to automatically select relevant input

variables from large datasets.

2. Model Selection: Employs Bayesian Optimization and comparative

methods to pick the best-performing network configuration.

3. Implementation: Built as a Java framework, designed to be extensible for

future neural network types.

4. Validation: Demonstrated on real-world applications like weather

forecasting, greenhouse modeling, etc.

Reported
results /
products /
effects:

• A fully automated framework (AGFACAND) that takes raw data as input

and returns a trained ANN model with minimal user intervention.

• Empirical evidence that the framework can match or outperform manually

tuned models.

• Broad applicability across multiple domains (time series forecasting, system

modeling, etc.).

What are the
contributions /
conclusions?

1. Full Automation: From feature selection to model deployment.
2. Multi-Network Support: MLP, ELM, SVR, and potential future

expansions.

3. Ease of Use: Designed for non-experts, with minimal required

configuration.

4

Your
evaluation of
the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance
of this
research with
respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

4. Practical Impact: Validated through diverse use cases, showing that

automated ANN configuration can be both robust and efficient.

• Significance: Automating ANN configuration is highly beneficial, especially

for non-specialists.

• Validity: The author substantiates claims with comparative experiments and

real-life examples; the methodology seems thorough.

• Novelty: While individual steps (feature selection, Bayesian Optimization)
are not new, their unified integration in a single, user-friendly framework is
noteworthy.

AGFACAND contributes significantly to the broader field of machine learning
automation (AutoML). Its focus on real-world applications (including
greenhouse climate modeling) highlights its practical viability. It may
encourage further research on automated approaches that reduce the manual
overhead of network design and tuning.

For a thesis focused on greenhouse climate management—especially one
requiring predictive, multi-objective optimization—AGFACAND could
serve as a vital component for automatically building and fine-tuning ANN-
based climate models. By streamlining the creation of predictive models, I
could integrate AGFACAND with my greenhouse control logic. This would
help:

1. Feature Selection: Quickly identify the most critical variables (e.g.,

temperature, humidity, CO₂, weather forecasts).

2. ANN Type Comparison: Seamlessly compare different network

architectures to model the greenhouse environment.

3. Hyperparameter Optimization: Automate the search for the best model

settings, potentially reducing energy or resource consumption.
4. Continuous Adaptation: As conditions change seasonally, the

framework’s extensibility could re-tune or retrain models periodically with
minimal effort.

Hence, the methods described in AGFACAND align strongly with my thesis’s
objective of using data-driven, proactive control strategies in greenhouse
management.

Paper 3: Climate control software integration with a greenhouse enviromental
control computer [4]

Title of the
paper:

Climate control software integration with a greenhouse enviromental control
computer

5

Name(s) of
author /
authors:

Jesper Mazanti Aaslyng
Niels Ehler
Lene Jakobsen

Motivation for
doing the
research:

The motivation for this paper is to make interfaces from an existing system of
IntelliGrow [2] to open up the general system of environmental control
computers (ECCs) to others and make integrations across the board.

The research
problem /
objective:

Is this
research
original?

Develop an interface for connecting a new application with the standard ECC.

The act of making interfaces is certainly not original but doing so for an ECC
in a non-vendor specific way that leads to broader adoptability might not have
been generally available before.

Chosen
research
approach /
methodology:

The study designed BipsArch, a six-layer system that combines greenhouse
control computers with advanced software such as IntelliGrow. It successfully
streamlined climate management over four months of testing at multiple
greenhouses through a vendor agnostic interface and real time data handling.

Reported
results /
products /
effects:

The project was generally very successful showing integrations with other
systems and databases as expected. It managed 45 climate inputs and 8 set
points reliably.

What are the
contributions /
conclusions?

The contributions of the paper would be that it tries successfully to make an
interface that opens up the ECC to broader audiences possibly adding to
scientific endeavors.

Evaluation showed good results but there was slight disruptions from the
database that shows on the results also. This however was negligible as it was
due to the database and not the interface. Better decisions on database would
resolve this.

As this shows interoperability it would be highly relevant as it means more
research and cross development can be conducted instead of only vendor
specific configurations.

Your
evaluation of
the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance
of this
research with
respect to the
field in
general:

Your
conclusion on

Again because of interoperability this might be use full for my studies as i
might end op using such an interface to connect with an ECC device.

6

the relevance
of this
research with
respect to your
own research:

Paper 4: Cost-efficient light control for production of two campanula species [5]

Title of the
paper:

Name(s) of
author /
authors:

Cost-efficient light control for production of two campanula species

Katrine Heinsvig Kjaer, Carl-Otto Ottosen, Bo Nørregaard Jørgensen

Motivation for
doing the
research:

The authors wanted to reduce electricity cost within greenhouses utilizing
weather forecasts and prices of electricity while having good quality in plant
health and production still.

The research
problem /
objective:

The objective was to evaluate a light control system that controls the
supplemental lighting in the greenhouse based on cost effectiveness while still
having food plant products.

Is this
research
original?

Using multi objective optimization to enhance cost of electricity and plant
product using weather forecast and the light need of the plant is somewhat
unique

Chosen
research
approach /
methodology:

A system using this multi objective optimization was used to test four
configurations in a greenhouse and on two species of campanula. This
combined the forecasted solar irradiance along with predefined light
requirements and electricity cost to optimize these four configurations

The results showed 25% cost reduction in electricity while using the system
without significant negative impact on the plants

Reported
results /
products /
effects:

What are the
contributions /
conclusions?

It was shown in the study that supplement light systems that are optimized
based in weather forecast and dynamic pricing can lower the cost of energy
while maintaining plant quality.

The results effectively achieved cost efficiency during spring. There was some
limitations that shows that there might be some importance for continuous
light under low natural light in certain conditions.

Your
evaluation of
the reported
results with
respect to the
research
objective:

7

This research is highly relevant for sustainable greenhouse operations,
providing a pathway for integrating cost-effective strategies into horticultural
practices globally.

This paper looks at multi objective optimization which also is an what my
study is looking into. Therefore its quite usable to have the insight into those
integrations that’s made beforehand.

Your
conclusion on
the relevance
of this
research with
respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

Paper 5: Enhancing State-of-the-Art Multi-Objective Optimization Algorithms
by Applying Domain-Specific Operators [6]

Title of the
paper:

Name(s) of
author /
authors:

Enhancing State-of-the-Art Multi-Objective Optimization Algorithms by
Applying Domain-Specific Operators

Seyyedeh Newsha Ghoreishi, Jan Corfixen Sørensen, Bo Nørregaard Jørgensen

Motivation for
doing the
research:

The research aimed to improve the convergence speed of multi-objective
evolutionary algorithms (MOEAs) for dynamic problems, particularly
greenhouse climate control, by introducing domain-specific operators.

The research
problem /
objective:

The research problem of this paper is to enhance performance of MOEAs for
dynamic optimization. This is done by integrating domain-specific knowledge
that enables faster convergence without sacrificing diversity or quality.

Is this
research
original?

Chosen
research
approach /
methodology:

Reported
results /
products /
effects:

MOEAs have been used before and have also been studied thoroughly. This
study however is novel as it applies domain-specific operators to a real-world
dynamic greenhouse control system.

The algorithm develop using domain specific initialization, mutation, and
crossover operators was tested against three algorithms that acts as baseline to
monitor performance on the greenhouse climate optimization.

As compared to NSGAII, ε-NSGAII, and ε-MOEA, CONTROLEUM-GA
converged faster with better diversity, finding high quality solutions with fewer
generations.

8

What are the
contributions /
conclusions?

The study has thus been able to demonstrate that incorporating domain
knowledge into MOEAs does improve their efficiency in dynamic
environments and can be employed as a reference in future applications.

The experiments’ results support the hypothesis that employing domain-
specific operators is advantageous for decreasing the convergence time without
compromising the solution quality and diversity, but the scalability of this
approach for a variety of applications is an open question.

Since this is a well studied area this already have been utilized in the filed of
horticulture and data-driven optimization.

Since my project is based on the already existing work of Dynagrow[1] then ill
naturally have to use it for my project also.

Your
evaluation of
the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance
of this
research with
respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

Paper 6: Advancements in smart thermostat technology for enhanced HVAC
Energy management[7]

Title of the
paper:

Name(s) of
author /
authors:

Advancements in Smart Thermostat Technology for Enhanced HVAC Energy
Management

Ireneo C. Plando, Jr.

Motivation for
doing the
research:

The motivation for this paper is to examine the potential for better performance
of energy management in context to Heating, ventilation and Air conditioning
(HVAC).

The research
problem /
objective:

The objective is to find potential ways of improving energy management for
better cost effectiveness and user comfort through literature review and
empirical analysis.

9

Is this
research
original?

Chosen
research
approach /
methodology:

Reported
results /
products /
effects:

Since this is a journal on that combines literature and empirical analysis id say
that this isn’t original as it only summarises existing work. It highlights the a
path potential path from existing research.

The approach is using literature review as already mentioned and also
empirical analysis. This is to gain a holistic overview of the advancements in
smart thermostats.

The reported results are as follows:

1. For commercial uses there are 30% improvements on energy savings.
2. For residential settings there was 25%.
3. User satisfaction is generally above 85%

Quantitative energy savings are about 3 and 7 percent points lower.

What are the
contributions /
conclusions?

Its evident that the current field of energy saving and user comfort in HVAC
systems have come far with above 20% general energy savings while
maintaining high user satisfaction in comfort and ease of use. But there is still
much room for further research to bridge existing gaps.

These results does show potential for further enhancements in regards to
energy saving and user comfort. As 25% and 30% is significant enhancement
it shows already prominent results.

This doesn’t show much prospect in the field of horticulture as the use of ml,
advanced algorithms and data analytics are already used for energy
optimization and good plant produce.

Your
evaluation of
the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance
of this
research with
respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

This study was supposed to sum of the advancements in smart heating to see
what the current state was for potential use and inspiration for my study in pro-
activeness in climate computers for horticulture use. As this focuses more on
user satisfaction and comfort i dont think that this is much use for my field as it
needs more fine grained control. Things i can take from this paper is the
importance of data analytics for ml models to produce substantial better
results.

10

Paper 7: A Novel Approach for Monitoring of Smart Greenhouse and Flowerpot
Parameters and Detection of Plant Growth with Sensors [8]

Title of the
paper:

Name(s) of
author /
authors:

A Novel Approach for Monitoring of Smart Greenhouse and Flowerpot
Parameters and Detection of Plant Growth with Sensors

Pinar Kirci and Erdinc Ozturk and Yavuz Celik

Motivation for
doing the
research:

The study was conducted to construct a smart greenhouse that uses sensor data
to optimize plant yield and save energy costs at the same time but with
multiple plants in the same growing area.

The research
problem /
objective:

The objective of the research was to:

1. Create a prototype of a smart greenhouse
2. Solve multi-objective problems with data-driven approaches
3. Grow multiple plants types in the same growing area / greenhouse.

Is this
research
original?

The study states that this is a novel approach. As the topic of multi-objective
problem solving with data-driven approaches isnt new, and similar study on
same growing area with different plants also immerge at the same time it not
that original. Although combining it is original, but not novel.

Chosen
research
approach /
methodology:

The study uses previous work as a stepping stone with literature review and
prototyping to achieve the goal. Various sensors and inexpensive hardware has
been used to try an gather enough data for the plants health to make the system
responsive and optimize plant health.

Reported
results /
products /
effects:

The results shows that 3 / 4 plants produced better in the smart greenhouse and
had better health all together where only one produced better outside but had
larger leafs in the smart greenhouse. Less water usage was also obtained in the
smart greenhouse. Study doesn’t mention energy usage.

What are the
contributions /
conclusions?

The conclusion shows that the there is real potential for their approach as it in
general produced better. They state that by using smart technologies in
agriculture greenhouses this might yield more produce and help mitigate the
growing problems associated with climate changes and geopolitical issues.

Your
evaluation of
the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance

As the results mainly use water consumption and crop yield to measure by I’m
left wondering what about energy consumption. The use of multi-objective
problem statements and trying to solve this with data-driven approaches is not
novel, but adding more plants in same greenhouse and adding smart
capabilities might be. Information about the data-driven approaches was also
lacking. Even though it might not be novel it still shows promise for the smart
aspects of greenhouses.

The contribution to the field might be negligible as information about the data-
driven approaches was very scares. It still shows that there is promise for using

11

of this
research with
respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

smart technologies in the field although addition in this paper might not be
much.

There isn’t much to use from this but aspects such as multi-objective problem,
data-driven and digital twins indicate that this is indeed the way that I should
also be thinking. Especially with the concept of digital twins. This area could
maybe be used for visual representation and simulation without having to grow
my own. Especially the simulation part is important for this as visualizing it
might now be entirely necessary.

Paper 8: Deep Learning Models for Health-Driven Forecasting of Indoor
Temperatures in Heat Waves in Canada: An Exploratory Study Using Smart
Thermostats[9]

Title of the
paper:

Name(s) of
author /
authors:

Deep Learning Models for Health-Driven Forecasting of Indoor Temperatures
in Heat Waves in Canada: An Exploratory Study Using Smart Thermostats

Jasleen Kaur and Gurjot Singh and Arlene Oetomo and Navneet Kaur and
Plinio P. Morita

Motivation for
doing the
research:

Because of the extreme heat that has outsourced over that last few years
causing significant risk to elderly people in Canada the team wanted to
investigate if deep learning could prove resourcefull

The research
problem /
objective:

The team chose to look into utilizing Deep learning for predicting indoor
temperatures data to forecast potential proactive warnings based on extreme
heat scenarios and possibly take action on this.

Is this
research
original?

Chosen
research
approach /
methodology:

Reported
results /
products /
effects:

Utilizing deep learning for proactive temperature forecasting in this field is
novel as is give super insight full predictions to potential actors that could take
responsible action to save lives.

Utilizing exiting knowledge of deep learning the team uses ETL (Extract,
Transform, Load, Analyze) to gives a solid ground to how solid foundation for
the study.

The results show a super effective model that mirrors the true temperature
trend closely and does so with minimal training. The model took
approximately 11.46 minutes to train compared to 2.65 hours. That is very
significant.

12

What are the
contributions /
conclusions?

As the model is easy to train and has high accuracy this shows a real world use
case to might be feasible for consultation for the Canadian government and
possibly others.

The results seem to be very usable and feasible. The training time seems to be
good as it utilizes cloud computing and thus cant be enhanced much for this
specific approach.

In the filed of horticulture this doesn’t show much. But the prospect of using
deep learning for temperature predictions could be useful in the sense that it
could show added performance for predictability and also plant growth
optimization.

As my project doesn’t has the aspect of using deep learning this doesn’t show
me much. But it does show me that deep learning might be a useful aspect for
the project as its highly likely to produce good models (if deep learning is
done correctly and the right resources are present)

Your
evaluation of
the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance
of this
research with
respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

Paper 9: Energy Efficiency and Economic Evaluation in Tomato Production: A
Case Study from Mersin Province in the Mediterranean Region[10]

Title of the
paper:

Name(s) of
author /
authors:

Energy Efficiency and Economic Evaluation in Tomato Production: A Case
Study from Mersin Province in the Mediterranean Region

Yelmen B., Sahin H.H., Cakir M.T.

Motivation for
doing the
research:

The motivation is that because Turkey produces a large amount of tomatoes
both in greenhouses and in open fields wether one or the other is more energy
efficient.

The research
problem /
objective:

The research’s motivation is to look at energy consumption comparison
between open field and greenhouses to look into wich is more energy
consuming and by how much.

13

Although similar approaches has been done comparing open field and
greenhouse to each other this study takes it a step further by using localized
data in comparison.

The Team gathered data for analysis first. Then did energy analysis to compare
energy usage and then economic analysis.

In general the reported results show that open field growing is less energy
consuming although greenhouse grown sell for more.

The study concludes that even though open field growing is less energy
consuming that for greenhouses there is a big advantage to reduce the overall
energy consumption as it still sells for more but striking balance with energy
consumption and production could lower the impact overall.

The results seems to be very solid with good verifications also. They highlight
great the differences between the open field and the greenhouse produce.

This definitely show that there is a need to lower the energy consumption for
greenhouses as this would have impact on both environmental factors as it
emits less but also striking the balance would still produce good yield for same
profit.

This is a good base for why optimizing the climate computer would be
beneficial. As i cant reduce diesel or energy sources optimizing for energy
efficiency would be impact full for profit and the environment

Is this
research
original?

Chosen
research
approach /
methodology:

Reported
results /
products /
effects:

What are the
contributions /
conclusions?

Your
evaluation of
the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance
of this
research with
respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

14

Paper 10: Energy Consumption Prediction of a Greenhouse and Optimization of
Daily Average Temperature[11]

Title of the
paper:

Name(s) of
author /
authors:

Motivation for
doing the
research:

The research
problem /
objective:

Is this
research
original?

Chosen
research
approach /
methodology:

Reported
results /
products /
effects:

What are the
contributions /
conclusions?

Your
evaluation of
the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance
of this
research with

Energy Consumption Prediction of a Greenhouse and Optimization of Daily
Average Temperature

Yongtao Shen,Ruihua Wei,Lihong Xu

Because that greenhouses have such as high energy footprint there is a need to
look into how to handle this effect.

The study aims to look into energy consumption and predicting the total
energy use, using three optimization algorithms and verifications to figure out
how high production can be maintained while gaining insight into energy
consumption and taking action from that.

The research seems to be original in using three distinct algorithms to do
comparison and using these results to predict energy consumption with
relatively accurate forecasting.

The research uses mathematical modeling to model out the equations needed
for the parameters. Then they use optimization algorithms to optimize the
model for near realtime data and then validates with real world data.

A study is presented, which offers a validated, real world applicable model for
predicting and optimising greenhouse energy use and it is shown that by
employing sound climate control strategies, energy efficiency and therefore
economic viability can be greatly improved.

The study successfully presents a data-driven, energy-efficient approach to
greenhouse heating method. The research presents a scalable and practical
solution for sustainable agriculture through the use of predictive modeling and
optimization algorithms and the validation of these in real-world conditions.

This research is relevant to energy optimized greenhouse climate control since
it presents a novel, data intelligent approach to climate control. Integrating
machine learning based optimization with real world energy modeling makes
the model both scientifically valid and practically impactful. Moreover, testing

15

in different greenhouse conditions and exploring integration with renewable
energy may enhance its scalability and feasibility.

This research seems like a larger implementation but aspects of it could be
insightful and useful.

respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

Paper 11: Optimization of energy ratio, benefit to cost and greenhouses gasses
using metaheuristic techniques (genetic and particular swarm algorithms) and
data envelopment analysis: Recommendations for mitigation of inputs
consumption (a case crop: edible onion) [12]

Title of the
paper:

Name(s) of
author /
authors:

Optimization of energy ratio, benefit to cost and greenhouses gasses using
metaheuristic techniques (genetic and particular swarm algorithms) and data
envelopment analysis: Recommendations for mitigation of inputs consumption
(a case crop: edible onion)

Behzad Elhami, Mohamoud Ghasemi Nejad Raeini, Morteza Taki, Afshin
Marban, Mohsen Heidarisoltanabadi

Motivation for
doing the
research:

This study looks at ways of overcoming problems of energy inefficiency, high
costs and emissions by applying DEA, MOGA and MOPSA to achieve
sustainability and optimum economic performance in onion farming.

The research
problem /
objective:

This research contributes to filling the gap between energy efficiency,
economic sustainability, and environmental impact in agriculture through a
data-driven optimization approach.

Is this
research
original?

Chosen
research
approach /
methodology:

Reported
results /

Multi objective algorithms are integrated to tailor energy, cost and emissions
in onion farming for the first time. The real world application and findings of
the research are also quite practical, thus it offers potential for sustainable
agriculture.

The study uses DEA to evaluate farm efficiency and optimizes energy use,
costs, and emissions using MOGA and MOPSA. Greenhouse gas emissions
(CO₂, CH₄, N₂O) are quantified and real world data is integrated with
MATLAB and Python to develop a scalable, sustainable agriculture
framework.

In the study, using DEA, MOGA, and MOPSA, energy, cost, and emission
optimization in onion production was achieved. Cut energy by 48.63%, costs
by 63.12%, and GHG emissions by 47%, while tripling the benefit-to-cost

16

products /
effects:

ratio. MOPSA increased energy use and emissions, but DEA had moderate
gains.

What are the
contributions /
conclusions?

• MOGA optimised energy, cost, and emissions very well which led to

reduced consumption and enhanced the best and environmental results.

• DEA offered mild optimization that led to rather small, but still quite

noticeable enhancements.

• MOPSA was ineffectual, worsening energy and emissions despite improving

economy.

• Metaheuristic algorithms are not guaranteed to be optimal, and other

methods may be more appropriate.

In this research, energy, cost and emissions are integrated and optimized to
prove the effectiveness of MOGA over DEA and MOPSA. The authors also
propose a novel approach for selecting strategies that considers multiple
factors and can, therefore, be useful in the context of sustainable agriculture

This research contributes to sustainable agriculture by showing how MOGA
can be used effectively to optimize onion production. Its comprehensive
approach enhances the state-of-the-art by offering practical solutions for
energy efficiency, cost reduction and emission control.

In this research, they have optimized energy use, costs, and emissions this
complements my work on data-driven greenhouse climate control. Multi-
objective algorithms enhance efficiency and sustainability. Integrating these
methods will refine predictive climate models for better resource management.

Your
evaluation of
the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance
of this
research with
respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

Paper 12: Prediction of Greenhouse Indoor Air Temperature Using Artificial
Intelligence (AI) Combined with Sensitivity Analysis [13]

Title of the
paper:

Name(s) of
author /
authors:

Prediction of Greenhouse Indoor Air Temperature Using Artificial Intelligence
(AI) Combined with Sensitivity Analysis

Pejman Hosseini Monjezi, Morteza Taki, Saman Abdanan Mehdizadeh, Abbas
Rohani, Md Shamim Ahamed

17

Motivation for
doing the
research:

The research was carried out in order to solve the problems of precise
prediction and management of the greenhouse indoor air temperature. The aim
was to establish the possibility of applying AI based models such as RBF,
SVM and GPR in the estimation of indoor air temperature in an even-span
polycarbonate greenhouse.

The research
problem /
objective:

The research problem is to build precise machine learning models that can
estimate the indoor air temperature of an even-span polycarbonate greenhouse.
The objectives of the research are as follows:

• AI based models for estimating the indoor air temperature of the greenhouse

are investigated.

• The performance of the models is compared and the best performing model

is identified.

• The selected model is optimized by tuning the input parameters and the

dataset size.

• The spread factor and the number of neurons in the hidden layer are

investigated in relation to the accuracy of the model.

Is this
research
original?

The research compares the performance of three different artificial intelligence
based models for predicting the indoor air temperature of a greenhouse where i
would mean it is original.

Chosen
research
approach /
methodology:

This study looks into exploring the use of AI–based models for accurate and
optimal prediction of greenhouse indoor air temperature. This paper looks at
the problem of exact temperature control as a critical factor that can improve
plant yield and resource utilization.

Reported
results /
products /
effects:

What are the
contributions /
conclusions?

The study focuses on AI based models to predict greenhouse indoor air
temperature to increase the accuracy and efficiency. This is because precise
temperature control is important in plant growth and energy management
hence accurate forecasting is crucial for proper climate control.

The main contributions or conclusions from the research are as follows:

• The RBF model is the best in predicting the indoor air temperature of the

greenhouse.

• The accuracy of the RBF model is a function of the dataset size, the value of
the spread factor, the number of neurons in the hidden layer, and the type of
training algorithm.

• The RBF model can be applied to the development of a smart control system

for greenhouses to optimize energy use and improve crop production.

• Controlling the energy flow in a greenhouse using intelligent ANN models
may result in a substantial reduction in energy consumption and costs.

Your
evaluation of

To train RBF, SVM, and GPR AI models, environmental sensors gather
information on temperature, humidity, solar radiation, and wind speed. The

18

sensitivity analysis is performed on inputs; hence, RBF is further tuned by
adjusting the spread factor and neurons. In addition, this improves the
greenhouse air temperature forecasting.

This research is helpful in enhancing the domain of smart agriculture by
showing the efficiency of AI-based models in predicting the greenhouse indoor
air temperature which can then be used in the control systems to improve the
climate and increase the yield of crops.

In this regard, the research I have conducted is in harmony with my project of
creating a data-driven greenhouse climate control system. The effectiveness of
AI based models in predicting greenhouse indoor air temperature as found in
this study is directly applicable to my work and the development and
implementation of my climate control system.

the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance
of this
research with
respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

Paper 13: Optimal Solar Greenhouses Design Using Multiobjective Genetic
Algorithm [14]

Title of the
paper:

Name(s) of
author /
authors:

Optimal Solar Greenhouses Design Using Multiobjective Genetic Algorithm

Bahram Mahjoob Karambasti, Mohamad Naghashzadegan, Maryam Ghodrat,
Ghadir Ghorbani, Roy B. V. B. Simorangkir, and Ali Lalbakhsh

Motivation for
doing the
research:

The motivation is to improve the ineffective energy consumption in
greenhouses especially in the subtropical climate areas. The study was
conducted to find the best physical parameters of common greenhouses in the
northern Iran to maximize the solar energy harvesting in a year.

The research
problem /
objective:

The purpose of optimizing greenhouse design was to achieve the maximum
use of year round solar energy, due to low efficiency in subtropical climates.
The study was aimed at determining the optimal parameters for even-span,
modified arch and Quonset greenhouses. The aim was to improve solar gain in
winter without overheating in summer.

Is this
research
original?

A new approach uses multi objective genetic algorithms for optimization of
solar efficiency in greenhouse in all seasons. New design parameters and
thermal modelling are incorporated to enhance energy performance. This
framework is scalable, sustainable and cost effective for agricultural practices.

19

Chosen
research
approach /
methodology:

Multi-objective genetic algorithms optimized greenhouse design for year-
round solar efficiency. Thermal modeling and simulations determined ideal
structural parameters. The approach enhances energy use and adaptability
across climates.

A multi objective optimization framework balance solar gain in winter and
minimize overheating in summer to increase greenhouse energy efficiency.
Genetic algorithms are used to optimize the structures which reduce energy
consumption by 15-25% and improve light distribution. The model is general
for sustainable greenhouse farming and energy efficient agricultural
infrastructure.

• An optimal greenhouse design developed using genetic algorithms for year-

round solar efficiency.

• Improved energy use by 15-25% through better thermal performance.
• Simulated balanced solar capture to avoid overheating.
• Introduce Ellipse Aspect Ratio (Z/W) for improved light distribution.
• A scalable and adaptable model for a variety of climates and different types

of greenhouses.

• Boosted sustainability by enhancing energy efficiency and decreasing the

carbon footprint.

The research makes a useful contribution by introducing a systematic, data-
driven way to solar greenhouse design optimization. It is scientifically sound,
an original approach, and could have useful application to sustainable
agriculture. But further adding to its geographical scope, incorporating real-
time regulation, and looking at combined energy systems would increase its
relevance even more.

Through the integration of genetic algorithms, this research improves the
design of solar-efficient greenhouses with a reduction in energy consumption
of 15-25%. It improves the structural parameters to enhance the thermal
performance of the building throughout the climates. The results of this
research are useful for sustainable agriculture and the improvement of resource
management.

Reported
results /
products /
effects:

What are the
contributions /
conclusions?

Your
evaluation of
the reported
results with
respect to the
research
objective:

Your
conclusion on
the relevance
of this
research with
respect to the
field in
general:

Your
conclusion on
the relevance
of this
research with
respect to your
own research:

From this research, I could incorporate a genetic algorithm-driven
optimization to enhance my climate control system’s predictive modelling. Its
finding on the solar energy efficiency is consistent with my objective of cutting
down the greenhouse energy consumption while optimizing plant growth. The
methodology can be used as a starting point for improving multi-objective
optimization in my simulations. and so a genetic algorithm based on the parent
algorithms is proposed to solve the optimization problem of the model.

20

References

Bibliography
[1]

J. C. Sørensen., K. H. Kjaer., C.-O. Ottosen., and B. N. Jørgensen., “DynaGrow – Multi-
Objective Optimization for Energy Cost-efficient Control of Supplemental Light in
Greenhouses,” in Proceedings of the 8th International Joint Conference on Computational
Intelligence (IJCCI 2016) - ECTA, SciTePress,  2016, pp. 41–48. doi:
10.5220/0006047500410048.

[2]

J. Aaslyng, J. Lund, N. Ehler, and E. Rosenqvist, “IntelliGrow: a greenhouse component-based
climate control system,” Environmental Modelling & Software, vol. 18, no. 7, pp. 657–666,
2003.

[3] M. G. Wollsen, “A Generic Framework for Automatic Configuration of Artificial Neural

Networks for Data Modeling,” 202AD.

[4]

J. Aaslyng, N. Ehler, and L. Jakobsen, “Climate control software integration with a greenhouse
environmental control computer,” Environmental Modelling & Software, vol. 20, no. 5, pp. 521–
527, 2005.

[5] K. H. Kjaer, C.-O. Ottosen, and B. N. Jørgensen, “Cost-efficient light control for production of

two campanula species,” Scientia Horticulturae, vol. 129, no. 4, pp. 825–831, 2011.

[6]

[7]

[8]

[9]

S. N. Ghoreishi, J. C. Sørensen, and B. N. Jørgensen, “Enhancing state-of-the-art multi-objective
optimization algorithms by applying domain specific operators,” in 2015 IEEE Symposium
Series on Computational Intelligence,  2015, pp. 877–884.

J. Ireneo C. Plando, “Advancements in Smart Thermostat Technology for Enhanced HVAC
Energy Management,” International Journal of Advanced Research in Science, Communication
and Technology (IJARSCT), vol. 3, no. 2, pp. 882–887, Jul. 2023, doi: 10.48175/
IJARSCT-12388.

P. Kirci, E. Ozturk, and Y. Celik, “A Novel Approach for Monitoring of Smart Greenhouse and
Flowerpot Parameters and Detection of Plant Growth with Sensors,” Agriculture, vol. 12, p.
1705, Oct. 2022, doi: 10.3390/agriculture12101705.

J. Kaur, G. Singh, A. Oetomo, N. Kaur, and P. P. Morita, “Deep Learning Models for Health-
Driven Forecasting of Indoor Temperatures in Heat Waves in Canada: An Exploratory Study
Using Smart Thermostats,” Digital Health and Informatics Innovations for Sustainable Health
Care Systems, pp. 1999–2000, 2024, doi: 10.3233/SHTI240826.

[10] B. Yelmen, H. H. Şahin, and M. T. Çakır, “Energy Efficiency and Economic Analysis in Tomato

Production: A Case Study of Mersin Province in the Mediterranean Region,” Applied Ecology
and Environmental Research, vol. 17, no. 4, pp. 7371–7379, 2019, doi: 10.15666/
aeer/1704_73717379.

[11] Y. Shen, R. Wei, and L. Xu, “Energy Consumption Prediction of a Greenhouse and Optimization

of Daily Average Temperature,” Energies, vol. 11, no. 65, p. , 2018, doi: 10.3390/en11010065.

[12] B. Elhami, M. Ghasemi Nejad Raeini, M. Taki, A. Marzban, and M. Heidarisoltanabadi,

“Optimization of Energy Ratio, Benefit to Cost and Greenhouses Gasses Using Metaheuristic
Techniques (Genetic and Particle Swarm Algorithms) and Data Envelopment Analysis,”
Environmental Progress & Sustainable Energy, vol. 41, no. 6, p. e13889, 2022, doi: 10.1002/
ep.13889.

21

[13] P. Hosseini Monjezi, M. Taki, S. Abdanan Mehdizadeh, A. Rohani, and M. S. Ahamed,

“Prediction of Greenhouse Indoor Air Temperature Using Artificial Intelligence (AI) Combined
with Sensitivity Analysis,” Horticulturae, vol. 9, no. 853, p. , 2023, doi: 10.3390/
horticulturae9080853.

[14] B. M. Karambasti, M. Naghashzadegan, M. Ghodrat, G. Ghorbani, R. B. V. B. Simorangkir, and

A. Lalbakhsh, “Optimal Solar Greenhouses Design Using Multiobjective Genetic Algorithm,”
IEEE Access, vol. 10, pp. 73728–73740, 2022, doi: 10.1109/ACCESS.2022.3189348.

22
