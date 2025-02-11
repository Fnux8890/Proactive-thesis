Article
Prediction of Greenhouse Indoor Air Temperature Using
Artiﬁcial Intelligence (AI) Combined with Sensitivity Analysis

Pejman Hosseini Monjezi 1, Morteza Taki 1,*
and Md Shamim Ahamed 3,*

, Saman Abdanan Mehdizadeh 1

, Abbas Rohani 2

1 Department of Agricultural Machinery and Mechanization Engineering, Faculty of Agricultural Engineering

and Rural Development, Agricultural Sciences and Natural Resources University of Khuzestan,
Mollasani 6341773637, Iran

2 Department of Biosystems Engineering, Faculty of Agriculture, Ferdowsi University of Mashhad,

Mashhad 9177948974, Iran

3 Department of Biological and Agricultural Engineering, University of California, Davis, CA 95616, USA
* Correspondence: mtaki@asnrukh.ac.ir (M.T.); mahamed@ucdavis.edu (M.S.A.)

Abstract: Greenhouses are essential for agricultural production in unfavorable climates. Accurate
temperature predictions are critical for controlling Heating, Ventilation, Air-Conditioning, and
Dehumidiﬁcation (HVACD) and lighting systems to optimize plant growth and reduce ﬁnancial
losses. In this study, several machine models were employed to predict indoor air temperature in an
even-span Mediterranean greenhouse. Radial Basis Function (RBF), Support Vector Machine (SVM),
and Gaussian Process Regression (GPR) were applied using external parameters such as outside air,
relative humidity, wind speed, and solar radiation. The results showed that an RBF model with the
LM learning algorithm outperformed the SVM and GPR models. The RBF model had high accuracy
and reliability with an RMSE of 0.82 ◦C, MAPE of 1.21%, TSSE of 474.07 ◦C, and EF of 1.00. Accurate
temperature prediction can help farmers manage their crops and resources efﬁciently and reduce
energy inefﬁciencies and lower yields. The integration of the RBF model into greenhouse control
systems can lead to signiﬁcant energy savings and cost reductions.

Keywords: greenhouses; indoor air temperature; machine learning; sensitivity analysis; spread factor;
energy savings

1. Introduction

The global population’s persistent growth and the reduction in available arable land
have led to the rapid expansion of agricultural greenhouses. The most critical factors
that affect the growth conditions of greenhouse plants include indoor air temperature,
humidity, soil temperature, light intensity, and carbon dioxide concentration [1]. However,
predicting the internal conditions of a greenhouse accurately can be challenging, as they
are dependent on various external factors [2,3]. Under unusual circumstances, the natural
environment may not be suitable for optimal crop growth as parameters like temperature,
relative humidity, photosynthetically active radiation (PAR) level, carbon dioxide level, etc.,
affect plant development [3].

Greenhouses are artiﬁcially controlled enclosed spaces where the indoor climate is reg-
ulated by the structure, cover, and by support from Heating, Ventilation, Air-Conditioning,
and Dehumidiﬁcation (HVACD) and lighting systems. The greenhouse cover is a crucial
structural component that allows useful light spectrum (between 400 and 700 nm) to pass
through for photosynthetic activities. All greenhouses absorb solar energy, but solar green-
houses are designed to store some of the heat for use at night or on cloudy days in addition
to absorbing solar energy during daylight hours [4].

The advancement of automation and artiﬁcial intelligence has led to a signiﬁcant
increase in the use of smart greenhouses. These greenhouses are equipped with tools

Citation: Hosseini Monjezi, P.; Taki,

M.; Abdanan Mehdizadeh, S.;

Rohani, A.; Ahamed, M.S. Prediction

of Greenhouse Indoor Air

Temperature Using Artiﬁcial

Intelligence (AI) Combined with

Sensitivity Analysis. Horticulturae

2023, 9, 853. https://doi.org/

10.3390/horticulturae9080853

Academic Editors: Most

Tahera Naznin, Kellie Walters and

Neil Mattson

Received: 12 July 2023

Revised: 23 July 2023

Accepted: 25 July 2023

Published: 26 July 2023

Copyright: © 2023 by the authors.

Licensee MDPI, Basel, Switzerland.

This article is an open access article

distributed under

the terms and

conditions of the Creative Commons

Attribution (CC BY) license (https://

creativecommons.org/licenses/by/

4.0/).

Horticulturae 2023, 9, 853. https://doi.org/10.3390/horticulturae9080853

https://www.mdpi.com/journal/horticulturae

horticulturaeHorticulturae 2023, 9, 853

2 of 18

and systems that aim to enhance the quantity and quality of the products while mini-
mizing energy consumption [5]. The primary task of these devices is to use appropriate
control algorithms to intelligently manage the indoor climatic conditions, including hu-
midity, temperature, CO2, and lighting, with the aim of reducing and optimizing energy
consumption [6,7].

Modern greenhouses measure, display, and control various parameters that affect the
growth of greenhouse products, such as environmental temperature and humidity, light
intensity and duration, carbon dioxide level, soil temperature, and other factors. These
systems are based on complex control algorithms and installed with many sensors both
inside and outside the greenhouse to stabilize the greenhouse conditions in an optimal
state according to the momentary values of these parameters [8]. However, increasing the
use of sensors can lead to higher initial costs for the greenhouse and ultimately to higher
prices for the harvested products.

On the other hand, growers’ awareness of the upcoming conditions during the day can
lead to quicker reactions and better management of energy resources in the greenhouse [9].
Therefore, many studies have been conducted since the early 20th century to model the
greenhouse energy loads [10,11], as well as indoor parameters such as temperature [12],
humidity [13], light intensity [14], CO2 [15], etc. The basis for all these research studies
is the initial modeling of the greenhouse conditions based on external variables such as
temperature, humidity, wind speed, radiation level, etc. [16–18].

Agricultural systems like greenhouses are very complex and dynamic systems, which
makes physics-based modeling difﬁcult. While dynamic models have been increasingly
used for predicting the inside situation of agricultural greenhouses, they come with certain
disadvantages. One of the main drawbacks of using dynamic models is that they require a
signiﬁcant amount of input data, which can be challenging to obtain in real-world scenarios.
This is especially true for systems that involve complex, nonlinear interactions between
different variables, such as temperature, humidity, light intensity, and air ﬂow [6]. Another
limitation of dynamic models is that they are highly dependent on the accuracy of the input
data. Any errors or uncertainties in the input data can signiﬁcantly impact the accuracy of
the model’s predictions. Additionally, dynamic models require a high level of expertise in
both modeling and agricultural sciences to develop and apply effectively [19]. Artiﬁcial
Intelligence (AI) has become increasingly popular in agricultural studies due to its ability
to model complex variables, which is essential in accurately predicting greenhouse climatic
parameters and loads. The accurate prediction of microclimate parameters like temperature,
humidity, and light intensity plays a crucial role in optimizing crop yield and quality while
minimizing energy consumption and environmental impact [3,13,19].

Machine learning (ML) techniques have gained popularity in predicting greenhouse
microclimate variables due to their ability to handle high-dimensional and noisy data,
learn from historical data, and adapt to changing conditions, making them suitable for
dynamic greenhouse environments [20,21]. Among the most common ML approaches used
in greenhouse microclimate prediction are Artiﬁcial Neural Networks (ANNs) and Support
Vector Regression (SVR). ANNs are inspired by the structure and function of the human
brain and consist of multiple layers of interconnected nodes that can recognize patterns in
the data. ANNs have been successfully applied to predict various greenhouse microclimate
parameters, such as air temperature, relative humidity, and PAR level [22,23]. SVR is a type
of supervised learning algorithm that can handle both linear and nonlinear data, and has
been used to predict greenhouse parameters such as air temperature, relative humidity,
and soil moisture content [24–26].

Horticulturae 2023, 9, 853

3 of 18

Other ML techniques such as Decision Trees (DT), Random Forests (RF), and Gaussian
Processes Regression (GPR) have also been applied to greenhouse microclimate prediction
with promising results [27]. DTs can handle both categorical and continuous data and
can be used to predict discrete outputs such as crop yield or continuous outputs such
as temperature and humidity. RFs are an ensemble of decision trees that can improve
prediction accuracy by combining the outputs of multiple trees. GPRs are a probabilistic
model that can predict the uncertainty in the data and make probabilistic predictions [28].
In conclusion, AI and ML techniques have become essential in accurately predicting
greenhouse microclimate parameters. Accurate predictions can help optimize crop yield
and quality while reducing energy consumption and environmental impact. Figure 1
describes the modeling approaches used in greenhouse control and management. Further
research can explore ways to optimize these models for reducing initial costs and energy
consumption while minimizing the environmental impact of greenhouse production.

Figure 1. Application of modeling in greenhouse control and management [29].

Although ML methods have shown promising results in predicting greenhouse micro-
climate parameters, there are still some challenges that need to be addressed. One of the
main challenges is the availability of high-quality data and the robustness of the models.
ML models require a large amount of high-quality data to train and validate the models, but
collecting and preprocessing data from greenhouse environments is often time-consuming
and challenging. Another challenge is the interpretability of the ML models, which are
often considered black boxes, making it difﬁcult to understand how the models make
predictions. Therefore, developing interpretable ML models that can provide insights
into the underlying relationships between microclimate parameters and crop growth is
essential [23,29].

To ensure reliable and accurate predictions in smart greenhouses, ANN models must
be optimized for robustness across a range of environmental conditions and input variables.
Recent studies have reviewed the use of AI for predicting various environmental and other
variables in greenhouses, as summarized in Table 1.

Horticulturae 2023, 9, x FOR PEER REVIEW 3 of 19   and can be used to predict discrete outputs such as crop yield or continuous outputs such as temperature and humidity. RFs are an ensemble of decision trees that can improve pre-diction accuracy by combining the outputs of multiple trees. GPRs are a probabilistic model that can predict the uncertainty in the data and make probabilistic predictions [28]. In conclusion, AI and ML techniques have become essential in accurately predicting greenhouse microclimate parameters. Accurate predictions can help optimize crop yield and quality while reducing energy consumption and environmental impact. Figure 1 de-scribes the modeling approaches used in greenhouse control and management. Further research can explore ways to optimize these models for reducing initial costs and energy consumption while minimizing the environmental impact of greenhouse production.  Figure 1. Application of modeling in greenhouse control and management [29]. Although ML methods have shown promising results in predicting greenhouse mi-croclimate parameters, there are still some challenges that need to be addressed. One of the main challenges is the availability of high-quality data and the robustness of the mod-els. ML models require a large amount of high-quality data to train and validate the mod-els, but collecting and preprocessing data from greenhouse environments is often time-consuming and challenging. Another challenge is the interpretability of the ML models, which are often considered black boxes, making it difficult to understand how the models make predictions. Therefore, developing interpretable ML models that can provide in-sights into the underlying relationships between microclimate parameters and crop growth is essential [23,29]. To ensure reliable and accurate predictions in smart greenhouses, ANN models must be optimized for robustness across a range of environmental conditions and input varia-bles. Recent studies have reviewed the use of AI for predicting various environmental and other variables in greenhouses, as summarized in Table 1. Table 1. Application of Artificial Intelligence (AI) in greenhouse modeling. References Subject Statistical Indexes [30] Model predictive control via output feedback Neural Network for improved multi-window greenhouse ventilation control RMSE value was 2.450 °C [31] Deep-learning-based prediction on greenhouse crop yield RMSEs (gm−2) for 3 dataset was: 10.450, 6.760 and 7.400, respectively Horticulturae 2023, 9, 853

4 of 18

Table 1. Application of Artiﬁcial Intelligence (AI) in greenhouse modeling.

References

Subject

Statistical Indexes

[30]

[31]

[32]

[33]

[20]

[34]

[13]

[35]

[36]

Model predictive control via output feedback Neural
Network for improved multi-window greenhouse
ventilation control
Deep-learning-based prediction on greenhouse
crop yield
Energy utilization assessment of a semi-closed
greenhouse using data-driven model predictive control
Machine learning algorithms to assess the thermal
behavior of a Moroccan agriculture greenhouse
Forecasting air temperature on edge devices with
embedded AI
The use of Artiﬁcial Neural Networks for forecasting of
air temperature indoor a heated foil tunnel

Neural Network model for greenhouse
microclimate predictions

Evaluation of CFD and machine learning methods on
predicting greenhouse microclimate parameters with the
assessment of seasonality impact on machine
learning performance
Data-driven robust model predictive control for
greenhouse temperature control and energy
utilization assessment

RMSE value was 2.450 ◦C

RMSEs (gm−2) for 3 dataset was: 10.450, 6.760 and
7.400, respectively
RMSE value for MPC method was 0.330 ◦C and 0.360
◦C for winter and summer simulation, respectively

R2 was 0.940 with 5-fold cross validation method
RMSE was 0.289–0.402 ◦C and MAPE
reported 0.87–1.04%
RMSE value reported: 3.700 ◦C

MAE, RMSE, and R2 were calculated to equal 0.218 K,
0.271 K, and 0.999 for temperature, and to 0.339%,
0.481%, and 0.999 for relative humidity

R > 0.980 and nRMSE < 9%

RMSE was 0.320 ◦C and 0.600 ◦C for a
two-day simulation

Scope, Innovations and Structure

Accurately predicting greenhouse indoor climates is crucial for optimizing crop yield
and quality while minimizing energy consumption. The literature reviewed in this study
emphasizes the importance of investigating accurate methods for predicting the indoor
climate of greenhouses. ANN models offer a more data-driven approach that can capture
the non-linear relationships between input variables, such as light, humidity, and tem-
perature. To address this issue, this research aims to investigate the potential of several
AI-based models, including Artiﬁcial Neural Network with Radial Basis Function (ANN-
RBF), Support Vector Machine (SVM), and Gaussian Process Regression (GPR) to estimate
the indoor air temperature of an even-span polycarbonate greenhouse. The methodology
employed in this study is outlined in Section 2 of the paper, which includes the study
area, data collection process, and the AI methods used to predict the indoor climate of
the experimental greenhouse. Section 3 reports the scientiﬁc ﬁndings of the study. The
results of the RBF, SVM, and GPR model analyses are presented and compared with other
similar studies. The discussion section of the paper presents suggestions for using this
method in future greenhouse applications, including developing a smart control system for
greenhouses. This would enable the real-time monitoring and control of the indoor climate,
leading to more efﬁcient energy use and increased crop yield. In the ﬁnal part of the paper,
conclusions and recommendations are presented based on the results of the study. The
ultimate goal of this research and its future development is to enable smart control systems
of greenhouses, leading to long-term reduction in energy losses.

2. Materials and Methods
2.1. Case Study and Data Collection

This study aimed to investigate a method for predicting the indoor air temperature in
an even-span polycarbonate greenhouse at the Agricultural Sciences and Natural Resources
at the University of Khuzestan, located 35 km north of Ahvaz, Iran (Latitude: 31.593;
Longitude: 48.892) (Figure 2). Data were collected in August and September of 2022 from
an even-span greenhouse structure with east–west orientation. It is utilized as a dryer in

Horticulturae 2023, 9, 853

5 of 18

spring and summer and as a place for growing plants in autumn and winter, owing to the
speciﬁc climatic conditions of the region. The greenhouse has a total area of 17 m2, an air
volume of 57 m3, and was empty of any plants. Temperature and humidity data indoor and
outside the greenhouse were collected using temperature sensors (SHT 11 made by CMOS
with an accuracy of ±0.4 ◦C and ±3% for temperature and humidity, respectively). Solar
radiation data indoor the greenhouse was collected on a leveled surface using a TES1333R
solar meter, which can collect radiation data in the wavelength range of 400 to 1100 nm
with an accuracy of approximately 5%. Wind speed data were extracted from the data on
Soda Service (https://www.soda-pro.com, accessed on 2 July 2023). An overview of the
greenhouse and all the experimental devices is presented in Figure 3. For the purposes of
this research, it was assumed that the greenhouse was fully enclosed and that all windows
were closed during data collection. Also, the data were collected with a 5 min interval for
10 days in September–October 2022.

Figure 2. Geographical location of the study area.

Horticulturae 2023, 9, x FOR PEER REVIEW 5 of 19   17 m2, an air volume of 57 m3, and was empty of any plants. Temperature and humidity data indoor and outside the greenhouse were collected using temperature sensors (SHT 11 made by CMOS with an accuracy of ±0.4 °C and ±3% for temperature and humidity, respectively). Solar radiation data indoor the greenhouse was collected on a leveled sur-face using a TES1333R solar meter, which can collect radiation data in the wavelength range of 400 to 1100 nm with an accuracy of approximately 5%. Wind speed data were extracted from the data on Soda Service (https://www.soda-pro.com, accessed on 2 July 2023). An overview of the greenhouse and all the experimental devices is presented in Figure 3. For the purposes of this research, it was assumed that the greenhouse was fully enclosed and that all windows were closed during data collection. Also, the data were collected with a 5 min interval for 10 days in September–October 2022.  Figure 2. Geographical location of the study area. Horticulturae 2023, 9, 853

6 of 18

Figure 3. Even-span polycarbonate greenhouse (a,b) with all the experimental devices, including
solar power meter (c) and SHT11 temperature and humidity sensor (d).

2.2. Radial Basis Function (RBF) Model

The ANN model is a popular prediction method comprising a minimum of three
layers. The ﬁrst layer, known as the input layer, has a size that is equivalent to the number
of inputs in the model. In this approach, each input has an associated weight. The hidden
layer comprises multiple neurons that enhance the performance of the ANN model by
ensuring a sufﬁcient number of neurons are present in this layer. The number of neurons
in the output layer is equivalent to the network output. As the objective of this study
is to forecast greenhouse temperatures, the output layer is conﬁgured to have just one
neuron [26]. On the other hand, in the RBF method, each neuron in the hidden layer
operates based on a nonlinear activation function. During the training phase of the RBF
neural network, the bias factor is utilized to converge the network and achieve the global
minimum [22].

To identify the most suitable network structure, the range of the spread parameter

was varied from 0.1 to 1.00 in this study.

Several approaches exist for training a network and adjusting weights to minimize the
error in the RBF model. Among these methods, the backpropagation algorithm of errors
is one of the most commonly employed techniques, as noted by Bolandnazar et al. [26].
In this study, thirteen training functions were employed to train the models using the
backpropagation training algorithms, as illustrated in Figure 4.

Horticulturae 2023, 9, x FOR PEER REVIEW 6 of 19     (a) (b)   (c) (d) Figure 3. Even-span polycarbonate greenhouse (a,b) with all the experimental devices, including solar power meter (c) and SHT11 temperature and humidity sensor (d). 2.2. Radial Basis Function (RBF) Model The ANN model is a popular prediction method comprising a minimum of three lay-ers. The first layer, known as the input layer, has a size that is equivalent to the number of inputs in the model. In this approach, each input has an associated weight. The hidden layer comprises multiple neurons that enhance the performance of the ANN model by ensuring a sufficient number of neurons are present in this layer. The number of neurons in the output layer is equivalent to the network output. As the objective of this study is to forecast greenhouse temperatures, the output layer is configured to have just one neuron [26]. On the other hand, in the RBF method, each neuron in the hidden layer operates based on a nonlinear activation function. During the training phase of the RBF neural net-work, the bias factor is utilized to converge the network and achieve the global minimum [22]. To identify the most suitable network structure, the range of the spread parameter was varied from 0.1 to 1.00 in this study. Several approaches exist for training a network and adjusting weights to minimize the error in the RBF model. Among these methods, the backpropagation algorithm of er-rors is one of the most commonly employed techniques, as noted by Bolandnazar et al. [26]. In this study, thirteen training functions were employed to train the models using the backpropagation training algorithms, as illustrated in Figure 4. Horticulturae 2023, 9, 853

7 of 18

Figure 4. The types of training functions applied in the RBF model in this study [26].

In the ANN model, it is feasible to approximate any optimal continuous function
by incorporating a hidden layer with a sufﬁcient number of neurons, as suggested by
Rohani et al. [22]. Accordingly, a single hidden layer was employed in this study to develop
the RBF model. To estimate the greenhouse temperature, the performance of the RBF
method was evaluated by adjusting the number of neurons in the hidden layer from 3 to
35, and the optimal conﬁguration was selected. Previous research has shown that linear
transfer functions in the output layer of the neural network method can approximate
complex functions effectively [6]. So, linear transfer functions for the RBF method were
implemented in the output layer in this study.

2.3. Support Vector Machine (SVM)

The SVM model acts as a proper computational method because the SVM method
can solve the quadratic optimization problems. The basic idea behind an SVM is to ﬁnd a
hyperplane in a high-dimensional space that separates the data into different classes [26].
The hyperplane is chosen so as to maximize the margin between the classes, which is
deﬁned as the distance between the hyperplane and the closest data points from each
class [19]. During the training phase, the SVM algorithm learns the optimal hyperplane by
ﬁnding the set of parameters that minimizes the classiﬁcation error on the training data.
Once the hyperplane has been learned, it can be used to predict the class label of new,
unseen data points [19]. To make a prediction using an SVM model, the algorithm takes
the input data and maps it into the high-dimensional feature space used during training. It
then applies the learned hyperplane to the transformed data to obtain a score or decision
function. The sign of the decision function indicates the predicted class label: if the decision
function is positive, the data point is classiﬁed into one class, and if it is negative, it is
classiﬁed into the other class. The magnitude of the decision function also provides a
measure of conﬁdence in the prediction [22].

Horticulturae 2023, 9, x FOR PEER REVIEW 7 of 19    Figure 4. The types of training functions applied in the RBF model in this study [26]. In the ANN model, it is feasible to approximate any optimal continuous function by incorporating a hidden layer with a sufficient number of neurons, as suggested by Rohani et al. [22]. Accordingly, a single hidden layer was employed in this study to develop the RBF model. To estimate the greenhouse temperature, the performance of the RBF method was evaluated by adjusting the number of neurons in the hidden layer from 3 to 35, and the optimal configuration was selected. Previous research has shown that linear transfer functions in the output layer of the neural network method can approximate complex functions effectively [6]. So, linear transfer functions for the RBF method were imple-mented in the output layer in this study. 2.3. Support Vector Machine (SVM) The SVM model acts as a proper computational method because the SVM method can solve the quadratic optimization problems. The basic idea behind an SVM is to find a hyperplane in a high-dimensional space that separates the data into different classes [26]. The hyperplane is chosen so as to maximize the margin between the classes, which is de-fined as the distance between the hyperplane and the closest data points from each class [19]. During the training phase, the SVM algorithm learns the optimal hyperplane by find-ing the set of parameters that minimizes the classification error on the training data. Once the hyperplane has been learned, it can be used to predict the class label of new, unseen data points [19]. To make a prediction using an SVM model, the algorithm takes the input data and maps it into the high-dimensional feature space used during training. It then applies the learned hyperplane to the transformed data to obtain a score or decision func-tion. The sign of the decision function indicates the predicted class label: if the decision function is positive, the data point is classified into one class, and if it is negative, it is classified into the other class. The magnitude of the decision function also provides a measure of confidence in the prediction [22].   Horticulturae 2023, 9, 853

8 of 18

2.4. Gaussian Process Regression (GPR) Model

Gaussian Process Regression (GPR) is a powerful statistical modeling technique that is
used to model complex data distributions and make predictions based on noisy or incom-
plete data. It is a non-parametric approach that assumes a prior probability distribution
over the possible functions that could ﬁt the data and updates this distribution as new
data are observed [37]. In GPR, the prior distribution over functions is represented by
a Gaussian process, which is a collection of random variables that are jointly Gaussian
distributed. The covariance between any two points in the input space determines the
similarity between those points and is used to make predictions about the output variable.
The hyper parameters of the Gaussian process, such as the length scale and amplitude,
control the smoothness and variability of the functions in the prior distribution [38]. The
posterior distribution over functions is obtained by conditioning on the observed data and
is also a Gaussian process with updated mean and covariance functions. This allows for the
computation of predictive distributions, and uncertainty estimates for new input points.
GPR has many advantages over other regression techniques, such as its ability to model
nonlinear and non-parametric relationships, ﬂexibility in choosing the covariance function,
and ability to provide uncertainty estimates for predictions [21]. It has been successfully
applied in many ﬁelds, including engineering, ﬁnance, and biology. However, GPR can
be computationally intensive and may require a careful choice of hyper parameters and
covariance functions. It also assumes that the data are stationary and does not account for
non-Gaussian noise in the data [21].

The GPR model aims to learn from training data and perform well in extrapolating the
output distribution to unseen input locations [39]. In this context, the noise in the output
model accounts for the uncertainty introduced by factors other than the input variable x,
such as observational errors. This study assumes that the noise is additive, has a zero mean,
is stationary, and tends to be randomly dispersed [22]:

y = f (x) + ε and ε (cid:39) N(0, s2

noise)

(1)

where s2
noise is the variance of the noise. The Gaussian prior assumption enables the function
to be interpreted through the mean m(x) and covariance functions. As suggested by the
existing literature, the shape of the mean function is only signiﬁcant in unobserved regions
and is often assumed to be zero [21].

2.5. Performance Evaluation Criteria

Various performance metrics were employed in this study to evaluate the effectiveness
of the RBF, SVM, and GPR models in predicting the indoor temperature in a greenhouse.
These metrics include the Mean Absolute Percentage Error (MAPE), Root Mean Square
Error (RMSE), Total Sum of Squared Error (TSSE), and Efﬁciency Factor (EF). In this study,
the model exhibiting the highest accuracy is determined by achieving the lowest MAPE,
RMSE, and TSSE values, and the highest EF value [22].

3. Results and Discussion

This study aimed to predict the indoor air temperature of an even-span polycarbonate
greenhouse using three different machine learning models, namely RBF, SVM, and GPR.
The dataset comprised observations from an even-span polycarbonate greenhouse, with
four factors used as inputs: Outside Solar Radiation (Iout) (Wm−2), Outside Air Temperature
(Tout) (◦C), Outside Air Humidity (Rhout) (%), and Outside Wind Speed (Wout).

3.1. Climate Condition of the Study Area

Figure 5 shows the climate variations in the studied area, which can impact crop
growth and productivity. The summer months in this region are characterized by high
temperatures that can exceed 50 ◦C, which can limit the growth of some crops when grown
outside. While greenhouse cultivation can provide optimal conditions for plant growth, it

Horticulturae 2023, 9, 853

9 of 18

also requires energy-intensive cooling systems to maintain suitable temperatures for most
of the time in the study area. Additionally, the high levels of solar radiation in the region
can be beneﬁcial for plant growth; however, they can also lead to heat stress and damage to
crops if not adequately managed. The low wind speeds in this region can be advantageous
for greenhouse cultivation, as they create a stable environment for plant growth and reduce
the risk of physical damage to the greenhouse structure. However, high humidity levels
during the summer and winter months can increase the spread of plant diseases, which
can be a signiﬁcant challenge for greenhouse cultivation.

Figure 5. Outside solar radiation (A), air temperature (B), wind speed (C), and humidity (D) in the
studied region in a year.

3.2. Selection of the Best Perform Models

This section presents the performance of several models in predicting the indoor air
temperature of the greenhouse, with the best-performing model selected. Table 2 shows
the statistical metrics employed to assess the accuracy of the models in the training, test,
and overall phases, which include MAPE, RMSE, TSSE, and EF. The results indicate that
the RBF model outperformed the other models in predicting the greenhouse indoor air
temperature, achieving a MAPE index ranging from 1.19 to 1.30%. The RBF model showed
lower MAPE values in the training and test phases than the GPR and SVM models. On the
other hand, the GPR model also demonstrated good accuracy for predicting the indoor air
temperature, while the SVM model did not perform well in this study. Based on the results,
the RBF model was selected for further analysis and development in the remaining parts of

Horticulturae 2023, 9, x FOR PEER REVIEW 9 of 19   grown outside. While greenhouse cultivation can provide optimal conditions for plant growth, it also requires energy-intensive cooling systems to maintain suitable tempera-tures for most of the time in the study area. Additionally, the high levels of solar radiation in the region can be beneficial for plant growth; however, they can also lead to heat stress and damage to crops if not adequately managed. The low wind speeds in this region can be advantageous for greenhouse cultivation, as they create a stable environment for plant growth and reduce the risk of physical damage to the greenhouse structure. However, high humidity levels during the summer and winter months can increase the spread of plant diseases, which can be a significant challenge for greenhouse cultivation.   (A) (B)   (C) (D) Figure 5. Outside solar radiation (A), air temperature (B), wind speed (C), and humidity (D) in the studied region in a year. 3.2. Selection of the Best Perform Models This section presents the performance of several models in predicting the indoor air temperature of the greenhouse, with the best-performing model selected. Table 2 shows the statistical metrics employed to assess the accuracy of the models in the training, test, and overall phases, which include MAPE, RMSE, TSSE, and EF. The results indicate that the RBF model outperformed the other models in predicting the greenhouse indoor air temperature, achieving a MAPE index ranging from 1.19 to 1.30%. The RBF model showed lower MAPE values in the training and test phases than the GPR and SVM models. On the other hand, the GPR model also demonstrated good accuracy for predicting the indoor air temperature, while the SVM model did not perform well in this study. Based on the results, the RBF model was selected for further analysis and development in the remaining Horticulturae 2023, 9, 853

10 of 18

the study. It should be noted that the use of this model is crucial for accurately predicting
the greenhouse indoor air temperature and ensuring the optimal growth of crops.

Table 2. Greenhouse temperature prediction by RBF, SVM, and GPR models.

Train

Test

Total

Model

RBF
SVM
GPR

RMSE MAPE

TSSE

0.80
1.97
1.29

1.19
3.13
1.93

367
2207
949

EF

1.00
0.97
0.99

RMSE MAPE

TSSE

0.91
2.13
2.19

1.30
3.70
4.12

108
596
626

EF

0.99
0.97
0.97

RMSE MAPE

TSSE

0.82
2.00
1.50

1.21
3.24
2.34

474
2803
1575

EF

1.00
0.97
0.98

Ali and Hassanein [40], developed a Recurrent Neural Network (RNN) model with
long short-term memory to predict environmental parameters in greenhouses, speciﬁcally
for tomato production. The model exhibited high accuracy and demonstrated its ability
to predict future temperatures, achieving an RMSE value of 0.7. Similarly, Petrakis and
Kavga [13], implemented neural network models to forecast microclimates in greenhouses
located in Greece. The results indicated maximum errors of 0.88 K and 2.84% for modeled
temperature and relative humidity, respectively, while the coefﬁcients of determination
were both 0.99 for these parameters.

3.3. Input Parameters Optimization

In this step, a sensitivity analysis was performed to develop and improve the selected
RBF model by considering the effects of the outside environment’s temperature, humidity,
radiation, and wind speed on the input data. The sensitivity analysis can provide valuable
insights into the behavior of the system and help identify the critical variables that affect
the model’s accuracy. By incorporating the effects of these variables into the model, the
accuracy of the predictions can be enhanced, leading to improved performance and efﬁ-
ciency of predictions. The variables were evaluated individually and then as a group to
determine their impact on the accuracy of the RBF model. Table 3 presents the results of
the sensitivity analysis.

Table 3. Sensitivity analysis for prediction the indoor air temperature of greenhouse.

Variables

Train

Test

Total

EF

TSSE MAPE

RMSE

All except Rhout
All except Tout
All except Iout
All except Wout
All data

1.48
1.67
2.07
0.84
0.8

2.08
2.27
3.05
1.22
1.19

1236.62
1590.1
2419.2
398.44
366.52

0.99
0.98
0.97
1
1

EF

1.53
1.52
1.91
0.96
0.91

TSSE MAPE

RMSE

2.41
2.41
2.83
1.51
1.3

308.3
300.94
480.31
120.42
107.56

0.99
0.99
0.98
0.99
0.99

EF

1.49
1.65
2.04
0.86
0.82

TSSE MAPE RMSE

2.14
2.27
3.01
1.27
1.21

1545.01
1891.04
2899.51
518.86
474.07

0.99
0.98
0.97
0.99
1

The ﬁndings reveal that including all input variables as datasets for RBF model training
leads to greater accuracy. This ﬁnding suggests that all the input variables play a critical role
in predicting the greenhouse indoor air temperature and that the RBF model’s performance
can be improved by considering all the variables simultaneously.

According to the sensitivity analysis results presented in Table 3, all four input vari-
ables, namely outside temperature, humidity, radiation, and wind speed, will be utilized as
the primary dataset to train the RBF model in the subsequent steps. The inclusion of these
variables is expected to result in more accurate predictions of the indoor air temperature in
the greenhouse. Based on the outcomes displayed in Table 2, the combination of outside
temperature, humidity, solar radiation, and wind speed was selected as the input dataset
for the RBF model in the subsequent steps of the study. This decision was based on the
sensitivity analysis results, indicating that considering all four variables simultaneously

Horticulturae 2023, 9, 853

11 of 18

can enhance the accuracy of the RBF model in forecasting the indoor air temperature of
the greenhouse.

3.4. Optimization of Dataset Sizes

The size of the dataset used to train the RBF network can have a signiﬁcant impact on
the accuracy and performance of the model. Generally, larger datasets lead to more accurate
predictions and can help avoid overﬁtting, which occurs when the model memorizes the
training data and performs poorly on new data. However, using a large dataset can also
increase the computational complexity and training time of the model, which can be a
limiting factor in some applications. In contrast, using a small dataset can result in under-
ﬁtting, where the model fails to capture the complex relationships between the input and
output variables.

To determine the optimal dataset size for training the RBF network, a sensitivity
analysis can be performed by training the model with different dataset sizes and evaluating
its performance using various statistical metrics. This approach can help identify the
minimum dataset size required for accurate predictions, while also avoiding overﬁtting
and excessive computational complexity.

This section evaluates the impact of dataset size on the accuracy of the RBF model
by varying the size of the dataset and analyzing the resulting changes in the model’s
accuracy (Table 4).

Table 4. Predicting greenhouse indoor air temperature using all variables as inputs with varying
data sizes.

The Share
of Dataset

80
70
60
50

Train

Test

Total

EF

0.81
0.92
1.16
1.05

TSSE MAPE

RMSE

1.28
1.31
1.39
1.39

364
415
565
383

1
0.99
0.99
0.99

EF

1.02
0.85
1.13
0.99

TSSE MAPE

RMSE

1.55
1.29
1.45
1.45

147
152
339
339

0.99
0.99
0.99
0.99

EF

0.86
0.9
1.15
1.02

TSSE MAPE RMSE

1.33
1.3
1.59
1.42

511
568
920
722

1
0.99
0.99
0.99

The results indicate that the optimal dataset size for training the RBF model to predict
the indoor air temperature in the greenhouse is 80% of the total dataset. This implies that
the model achieves the highest accuracy when trained with 80% of the total dataset. The
MAPE index for predicting the indoor air temperature in the entire phases was 1.33%,
demonstrating that the RBF model can accurately predict the output. However, in the
standard mode, the best results may not always be achieved when 60% of the total data
are used for network training, as observed in this study. To ensure the highest accuracy of
the RBF model, the dataset size for training the model was ﬁxed at 80% in the subsequent
analyses. This study emphasizes the importance of selecting the optimal dataset size to
achieve the best results and avoid overﬁtting or under-ﬁtting.

3.5. Selection of Best Training Algorithm for RBF Model

The selection of the best training algorithm for the RBF neural network model can have
a signiﬁcant impact on the accuracy and performance. Different training algorithms can
vary in their convergence speed, computational complexity, and ability to avoid overﬁtting.
One commonly used training algorithm for RBF neural networks is the backpropagation
algorithm, which involves iteratively adjusting the weights and biases of the network to
minimize the error between the predicted and actual outputs. While the backpropagation
algorithm can be effective in training RBF models, it may also suffer from slow convergence
and the risk of getting trapped in local minima. Other training algorithms, such as the
Levenberg–Marquardt algorithm, can offer faster convergence and better generalization
performance by adjusting the learning rate based on the curvature of the error surface. The

Horticulturae 2023, 9, 853

12 of 18

Quasi–Newton algorithm can also be effective in training RBF models by approximating the
second derivative of the error function and adjusting the weights and biases accordingly.
In this study, the performance of 13 different training algorithms for the RBF neural
network model was evaluated and compared (Table 5). The results indicate that the
Levenberg–Marquardt algorithm (trainlm) achieved the lowest MAPE, RMSE, and TSSE,
as well as the highest EF at the total phase, indicating superior accuracy and performance
compared to the other algorithms evaluated. The Levenberg–Marquardt algorithm is a
popular training algorithm for RBF neural networks due to its ability to converge more
quickly than other algorithms such as the backpropagation algorithm, while also being
less prone to overﬁtting than more complex algorithms like the Bayesian regularization
algorithm. The use of the Levenberg–Marquardt algorithm in the next analysis is expected
to lead to the improved accuracy and performance of the RBF neural network model for
predicting greenhouse indoor air temperature, particularly when training on large datasets.
Castañeda-Miranda and Castaño [41] utilized a Multi-Layer Perceptron (MLP) Artiﬁcial
Neural Network (ANN) to predict greenhouse air temperature. The ANN was trained with
a Levenberg–Marquardt backpropagation algorithm, with the input parameters consisted
of outside air temperature and relative humidity, global solar radiation, wind speed, and
indoor relative humidity. The study reported a temperature forecast with 95% conﬁdence,
achieving a coefﬁcient of determination of 0.96 in winter and 0.95 in summer. Yue et al. [42],
proposed an improved Levenberg–Marquardt Radial Basis Function Neural Network (LM-
RBF) model to forecast greenhouse air temperature and humidity. Their model achieved a
maximum relative error of less than 0.5%.

Table 5. Optimizing training algorithm selection for predicting greenhouse indoor air temperature
using statistical indexes in train, test, and total phases.

Training
Algorithm

trainlm
trainbr
trainbfg
traincgb
traincgf
traincgp
trainrp
trainoss
traingdx
trainscg
traingda
traingdm
traingd

Train

Test

Total

RMSE MAPE

TSSE

0.80
0.84
1.07
1.30
1.34
1.32
1.34
1.41
1.54
1.62
1.71
2.63
5.50

1.19
1.31
1.55
1.82
1.86
1.91
1.98
2.00
2.22
2.41
2.47
4.06
10.28

367
401
646
963
1025
988
1019
1126
1348
1484
1658
3921
17151

EF

1.00
1.00
0.99
0.99
0.99
0.99
0.99
0.99
0.98
0.98
0.98
0.95
0.79

RMSE MAPE

TSSE

0.91
1.08
1.13
1.12
1.22
1.36
1.34
1.39
1.34
1.39
1.41
1.89
5.42

1.30
1.62
1.72
1.77
1.94
2.21
2.16
2.12
2.24
2.50
2.48
3.28
11.07

108
152
168
164
194
243
235
252
236
252
259
470
3851

EF

0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.98
0.82

RMSE MAPE

TSSE

0.82
0.89
1.08
1.27
1.32
1.33
1.34
1.41
1.51
1.58
1.66
2.51
5.49

1.21
1.37
1.59
1.81
1.87
1.97
2.01
2.03
2.22
2.43
2.47
3.91
10.43

474
552
815
1127
1219
1231
1254
1378
1584
1736
1917
4392
21003

EF

1.00
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.98
0.98
0.98
0.80
0.80

3.6. Optimization of Hidden Layer Neurons

The hidden layer of the RBF neural network model plays a critical role in transform-
ing the input data into a new space that is more suitable for linearly separable analysis.
Unlike linear models, the RBF model can handle nonlinear patterns in the input data by
transforming them into a higher-dimensional space through the hidden layer. The number
of neurons in the hidden layer is an important parameter that affects the model’s ability to
capture complex relationships between the input and output variables. Cover’s theorem
on the reparability of patterns suggests that nonlinear patterns in the input data can be
transformed into a higher-dimensional space to make them more linearly separable. So, the
number of neurons in the hidden layer should be greater than the number of input neurons
to increase the dimensionality of the transformed space and improve the model’s ability
to capture nonlinear relationships. Also, the optimal number of neurons in the hidden
layer depends on the complexity of the input and output data, as well as the degree of

Horticulturae 2023, 9, 853

13 of 18

nonlinearity in the relationships between them. A small number of neurons in the hidden
layer can lead to under-ﬁtting, where the RBF model is too simple and unable to capture
the complex relationships between the input and output variables. On the other hand,
a large number of neurons in the hidden layer can result in overﬁtting, where the RBF
model memorizes the training data and performs poorly on new data. To determine the
optimal number of neurons in the hidden layer, a sensitivity analysis can be performed by
training the RBF model with different numbers of neurons and evaluating its performance
using various statistical metrics. This approach can help identify the optimal number of
neurons that achieves the best balance between overﬁtting and under-ﬁtting and achieves
the best accuracy and efﬁciency. This study examines the impact of the number of neurons
in the hidden layer on the accuracy and performance of the RBF neural network model in
predicting the indoor air temperature of the greenhouse. The number of neurons in the
hidden layer was varied from 3 to 35, and the model’s performance was evaluated based
on the lowest error and highest accuracy (Figure 6). The ﬁndings demonstrate that the
optimal number of neurons in the hidden layer was 33. By ﬁxing the number of neurons in
the hidden layer to this value, the MAPE factor reduced to 1.3%, indicating a signiﬁcant
enhancement in the model’s accuracy and performance. Francik and Kurpaska [34] devel-
oped a three-layer Perceptron neural network with 10 neurons in the hidden layer, utilizing
temperature, wind speed, solar radiation, and forecast time as input parameters to forecast
temperature changes in a heated foil tunnel. The study achieved the lowest RMSE value
(3.7 ◦C) for the testing dataset.

Figure 6. Selecting the best number of neurons in hidden layer for RBF model based on two statistical
indexes; MAPE (A) and RMSE (B).

3.7. Effect of Spread Factor on the Efﬁciency of RBF Model

The spread factor is a crucial parameter in the RBF model that can signiﬁcantly impact
the efﬁciency and accuracy of the model. The spread factor determines the width of the
RBF kernel function, which affects the degree of overlap between the RBF functions and
the spatial distribution of the input data.

To determine the optimal spread factor for the RBF model, a sensitivity analysis
can be performed by training the model with different spread factors and evaluating its
performance using various statistical metrics. This approach can help identify the optimal
spread factor that balances between overﬁtting and under-ﬁtting and achieves the best
accuracy and efﬁciency. In general, the optimal spread factor depends on the characteristics
of the input data and the complexity of the relationships between the input and output
variables. For complex and highly nonlinear systems, a smaller spread factor may be more
appropriate, while simpler systems may require a larger spread factor. In this study, the
spread factor was varied from 0.1 to 1, and for each value, the MAPE, RMSE, TSSE, and EF
factors were computed (Figure 7). The results indicate that, by selecting 0.2 for spread factor,
the accuracy can considerable increase. This ﬁnding highlights the importance of selecting
an optimal spread factor to achieve the best results in RBF modeling. The MAPE and RMSE

Horticulturae 2023, 9, x FOR PEER REVIEW 13 of 19   neurons to increase the dimensionality of the transformed space and improve the model’s ability to capture nonlinear relationships. Also, the optimal number of neurons in the hid-den layer depends on the complexity of the input and output data, as well as the degree of nonlinearity in the relationships between them. A small number of neurons in the hid-den layer can lead to under-fitting, where the RBF model is too simple and unable to cap-ture the complex relationships between the input and output variables. On the other hand, a large number of neurons in the hidden layer can result in overfitting, where the RBF model memorizes the training data and performs poorly on new data. To determine the optimal number of neurons in the hidden layer, a sensitivity analysis can be performed by training the RBF model with different numbers of neurons and evaluating its perfor-mance using various statistical metrics. This approach can help identify the optimal num-ber of neurons that achieves the best balance between overfitting and under-fitting and achieves the best accuracy and efficiency. This study examines the impact of the number of neurons in the hidden layer on the accuracy and performance of the RBF neural net-work model in predicting the indoor air temperature of the greenhouse. The number of neurons in the hidden layer was varied from 3 to 35, and the model’s performance was evaluated based on the lowest error and highest accuracy (Figure 6). The findings demon-strate that the optimal number of neurons in the hidden layer was 33. By fixing the number of neurons in the hidden layer to this value, the MAPE factor reduced to 1.3%, indicating a significant enhancement in the model’s accuracy and performance. Francik and Kurpaska [34] developed a three-layer Perceptron neural network with 10 neurons in the hidden layer, utilizing temperature, wind speed, solar radiation, and forecast time as in-put parameters to forecast temperature changes in a heated foil tunnel. The study achieved the lowest RMSE value (3.7 °C) for the testing dataset.   (A) (B) Figure 6. Selecting the best number of neurons in hidden layer for RBF model based on two statis-tical indexes; MAPE (A) and RMSE (B). 3.7. Effect of Spread Factor on the Efficiency of RBF Model The spread factor is a crucial parameter in the RBF model that can significantly im-pact the efficiency and accuracy of the model. The spread factor determines the width of the RBF kernel function, which affects the degree of overlap between the RBF functions and the spatial distribution of the input data. To determine the optimal spread factor for the RBF model, a sensitivity analysis can be performed by training the model with different spread factors and evaluating its per-formance using various statistical metrics. This approach can help identify the optimal spread factor that balances between overfitting and under-fitting and achieves the best accuracy and efficiency. In general, the optimal spread factor depends on the characteris-tics of the input data and the complexity of the relationships between the input and output variables. For complex and highly nonlinear systems, a smaller spread factor may be more appropriate, while simpler systems may require a larger spread factor. In this study, the 1.300112233445369121518212427303336MAPE (%)Number of neurons in hidden layerTrainTest0.910.00.51.01.52.02.53.0369121518212427303336RMSE (ºC)Number of neurons in hidden layerTrainTestHorticulturae 2023, 9, 853

14 of 18

at the training and test phases are 1.30% and 0.91 ◦C, respectively. Figure 8 shows the
distribution of actual and predicted data (45-degree line) from the RBF model. It can be
concluded that the RBF model could predict the indoor air temperature of greenhouse with
high accuracy and can be used for climate controlling in smart greenhouses.

Figure 7. Selecting the best number of spread parameter for RBF model based on MAPE (A) and
RMSE (B) as evaluation metrics.

Figure 8. Comparing predicted by RBF model with actual values.

In a study, a hybrid artiﬁcial neural network (ANN) was utilized to predict freshwater
production in seawater greenhouses [43]. The study demonstrated that the ANN method is
highly accurate, with negligible differences between actual and predicted data. In another
study, machine learning algorithms were employed to predict indoor air temperature in
Moroccan agriculture greenhouses [33]. The results showed that all predictive models
performed well, with an R2 value greater than 0.9.

Table 6 shows the statistical properties of the data utilized in the training, test, and
overall stages of the selected RBF structure for predicting the indoor air temperature in
the greenhouse. The outcomes demonstrate that the differences between the minimum,
maximum, variance, and skewness of the actual and predicted data are negligible, which is
insigniﬁcant for practical purposes.

Horticulturae 2023, 9, x FOR PEER REVIEW 14 of 19   spread factor was varied from 0.1 to 1, and for each value, the MAPE, RMSE, TSSE, and EF factors were computed (Figure 7). The results indicate that, by selecting 0.2 for spread factor, the accuracy can considerable increase. This finding highlights the importance of selecting an optimal spread factor to achieve the best results in RBF modeling. The MAPE and RMSE at the training and test phases are 1.30% and 0.91 °C, respectively. Figure 8 shows the distribution of actual and predicted data (45-degree line) from the RBF model. It can be concluded that the RBF model could predict the indoor air temperature of green-house with high accuracy and can be used for climate controlling in smart greenhouses.   (A) (B) Figure 7. Selecting the best number of spread parameter for RBF model based on MAPE (A) and RMSE (B) as evaluation metrics.  Figure 8. Comparing predicted by RBF model with actual values. In a study, a hybrid artificial neural network (ANN) was utilized to predict freshwa-ter production in seawater greenhouses [43]. The study demonstrated that the ANN method is highly accurate, with negligible differences between actual and predicted data. In another study, machine learning algorithms were employed to predict indoor air tem-perature in Moroccan agriculture greenhouses [33]. The results showed that all predictive models performed well, with an R2 value greater than 0.9. Table 6 shows the statistical properties of the data utilized in the training, test, and overall stages of the selected RBF structure for predicting the indoor air temperature in the greenhouse. The outcomes demonstrate that the differences between the minimum, 1.30011223344500.20.40.60.81MAPE (%)Number of spread factorTrainTest0.910.00.51.01.52.02.53.000.20.40.60.81RMSE (ºC)Number of spread factorTrainTestTrain: y =0.99x+0.02, R² =0.99Test:  y =0.98x+1.02, R² =0.99203040506070203040506070Predicated data  Actual dataTrainTestBest fitHorticulturae 2023, 9, x FOR PEER REVIEW 14 of 19   spread factor was varied from 0.1 to 1, and for each value, the MAPE, RMSE, TSSE, and EF factors were computed (Figure 7). The results indicate that, by selecting 0.2 for spread factor, the accuracy can considerable increase. This finding highlights the importance of selecting an optimal spread factor to achieve the best results in RBF modeling. The MAPE and RMSE at the training and test phases are 1.30% and 0.91 °C, respectively. Figure 8 shows the distribution of actual and predicted data (45-degree line) from the RBF model. It can be concluded that the RBF model could predict the indoor air temperature of green-house with high accuracy and can be used for climate controlling in smart greenhouses.   (A) (B) Figure 7. Selecting the best number of spread parameter for RBF model based on MAPE (A) and RMSE (B) as evaluation metrics.  Figure 8. Comparing predicted by RBF model with actual values. In a study, a hybrid artificial neural network (ANN) was utilized to predict freshwa-ter production in seawater greenhouses [43]. The study demonstrated that the ANN method is highly accurate, with negligible differences between actual and predicted data. In another study, machine learning algorithms were employed to predict indoor air tem-perature in Moroccan agriculture greenhouses [33]. The results showed that all predictive models performed well, with an R2 value greater than 0.9. Table 6 shows the statistical properties of the data utilized in the training, test, and overall stages of the selected RBF structure for predicting the indoor air temperature in the greenhouse. The outcomes demonstrate that the differences between the minimum, 1.30011223344500.20.40.60.81MAPE (%)Number of spread factorTrainTest0.910.00.51.01.52.02.53.000.20.40.60.81RMSE (ºC)Number of spread factorTrainTestTrain: y =0.99x+0.02, R² =0.99Test:  y =0.98x+1.02, R² =0.99203040506070203040506070Predicated data  Actual dataTrainTestBest fitHorticulturae 2023, 9, 853

15 of 18

Table 6. Comparing the performance of RBF model with actual data at all phases of modeling.

Minimum Maximum Kurtosis

Skewness

Sum

Phases

Data

Average

Variance

Train

Actual
Predicted

49.51
49.50

145.98
146.40

Standard
Deviation

12.08
12.10

22.02
22.33

65.00
64.25

Predicted values = 0.99 × Actual values + (0.02) R2 = 0.99

Test

Actual
Predicted

49.32
49.38

161.77
156.29

12.72
12.50

22.00
22.33

63.00
63.58

Predicted values = 0.98 × Actual values + (0.1.02) R2 = 0.99

Total

Actual
Predicted

49.48
49.48

148.72
148.04

12.20
12.17

22.00
22.33

65.00
64.25

2.45
2.43

2.24
2.33

2.41
2.41

−0.77
−0.77

−0.67
−0.71

−0.75
−0.76

28073.90
28065.41

6460.87
6469.36

34534.77
34534.77

The best model results are obtained when the linear relationship between the actual
and predicted values has the highest coefﬁcient of determination, the narrowest width
from the origin, and a slope close to one. In this study, the RBF model exhibited a strong
correlation coefﬁcient in the training and testing phases, with regression relationships
having the smallest width from the origin and a slope close to one. Hence, this model is
considered the best for prediction. To further evaluate the RBF model, various statistical
tests were conducted in this study. The tests analyzed the average, variance, and statistical
distribution of the actual and predicted values by the RBF model in different stages of
training, testing, and overall. The null hypothesis for each test is the equality of mean,
variance, and statistical distribution of both data series:

(cid:40)

H0 : ya = yp
H1 : ya (cid:54)= yp

(cid:40)

and

H0 : σ2
H1 : σ2

ya = σ2
yp
ya (cid:54)= σ2
yp

and

(cid:26) H0 : da = dp
H1 : da (cid:54)= dp

(2)

At a signiﬁcance level of 95%, each hypothesis was tested using the p-value parameter.
If the calculated p-value for each stage exceeds 0.05, the null hypothesis cannot be rejected.
To compare the mean, variance, and statistical distribution, t-tests, F-tests, and Kolmogorov–
Smirnov tests were employed. Table 7 shows the p-values computed for all three stages
(training, test, and overall).

Table 7. Performance of optimized RBF model for greenhouse indoor air temperature prediction.

Phases

Training
Test
Total

Types of Statistical Analysis

Final Statistical Indexes of RBF Model

Average

Variance

Distribution

RMSE

MAPE

0.98
0.97
1.00

0.97
0.84
0.95

0.82
0.52
0.87

0.80
0.91
0.82

1.19
1.30
1.21

TSSE

366.52
107.56
474.07

EF

1.00
0.99
1.00

The results demonstrate that the mean, variance, and statistical distribution values
of the data obtained from the RBF model exhibit no signiﬁcant difference from the actual
values, indicating that this model can be utilized with high reliability.

4. Conclusions

Machine learning (ML) techniques have become increasingly important in model-
ing complex systems. ML enables more accurate and reliable predictions by leveraging
large datasets and capturing complex relationships between input variables and output
targets. In the context of predicting indoor air temperature, ML models can account for
various factors, such as outdoor temperature, humidity, solar radiation, and occupant
behavior, resulting in more comprehensive and holistic predictions. These predictions
can be beneﬁcial for optimizing energy consumption, improving indoor comfort and air

Horticulturae 2023, 9, 853

16 of 18

quality, and reducing greenhouse gas emissions. Furthermore, ML models can adapt to
changing conditions and learn from experience, making them ideal for predicting indoor
air temperature in dynamic environments.

The primary objective of this study was to develop accurate ML models for predicting
indoor air temperature in an even-span polycarbonate greenhouse using RBF, SVM, and
GPR models. The results of the study are presented as follows:

1.

2.

3.

The comparison of the three models revealed that the RBF model was the most ef-
fective in accurately predicting greenhouse temperature. The RBF model achieved
the lowest RMSE values during the training and test phases, at 0.80 ◦C and
0.91 ◦C, respectively.
The evaluation of the RBF model’s performance showed that the dataset size, value of
spared factor, number of neurons in the hidden layer, and type of training algorithm
signiﬁcantly impacted the output.
Accurate temperature prediction is crucial for achieving the goal of smart greenhouse
operation, and the high accuracy and reliability of the RBF model make it a valuable
tool for optimizing greenhouse management, improving time management, and
increasing crop yields. The performance results of this study indicate that integrating
artiﬁcial neural network (ANN) models into the control system can assist farmers in
building smart greenhouses.

Author Contributions: P.H.M.: Conceptualization, Methodology, Investigation; M.T.: Supervising,
Software, review & editing; S.A.M.: review & editing; A.R.: Advisor, Software, review & editing; M.S.A.:
Advisor; review & editing. All authors have read and agreed to the published version of the manuscript.

Funding: This research received no external funding.

Data Availability Statement: The datasets used and analyzed during the current study are available
from the corresponding author on reasonable request.

Acknowledgments: This study was supported by the Agricultural Sciences and Natural Resources
University of Khuzestan, Iran and the University of California Davis, Davis, USA. The authors are
grateful for the support provided by these Universities. This study is a part of an MSc thesis in the
Department of Agricultural Machinery and Mechanization Engineering, Agricultural Sciences and
Natural Resources University of Khuzestan, Iran.

Conﬂicts of Interest: The authors declare that they have no conﬂict of interest.

Nomenclature
MAPE
PSO
ANFIS
Trainbr
Trainbfg
Traincgb
Trainscg
Traincgf
Trainoss
Traincgp
Trainlm
Trainrp
Traingdx
Traingda
Traingdm
Traingd
AI
BP

Mean Absolute Percentage Error
Particle Swarm Optimization
An adaptive neuro-fuzzy inference system
Train by Bayesian regularization backpropagation
Train by BFGS quasi-Newton backpropagation
Train by Powell–Beale conjugate gradient backpropagation
Train by Scaled conjugate gradient backpropagation
Train by Fletcher–Powell conjugate gradient backpropagation
Train by One step secant backpropagation
Train by Polak–Ribiere conjugate gradient backpropagation
Train by Levenberg–Marquardt backpropagation
Train by Resilient backpropagation
Train by Gradient descent w/momentum and adaptive backpropagation
Train by Gradient descent with adaptive backpropagation
Train by Gradient descent with momentum backpropagation
Train by Gradient descent backpropagation
Artiﬁcial Intelligence
Back Propagation

Horticulturae 2023, 9, 853

17 of 18

DL
DNN
RMSE
ML
NN
R2
SA
SVR
RF
DT
EF
ANN
MLR
RBF
SVM

Deep Learning
Deep Neural Network
Root Mean Square Error
Machine Learning
Neural Network
Coefﬁcient of determination
Smart Agriculture
Support Vector Regression
Random Forests
Decision Trees
Efﬁciency Factor
Artiﬁcial Neural Network
Multiple Linear Regression
Radial Basis Function
Support Vector Machine

References

1.

2.

3.

4.

5.

6.

7.
8.

9.

Amini, S.; Taki, M.; Rohani, A. Applied improved RBF neural network model for predicting the broiler output energies. Appl. Soft
Comput. J. 2020, 87, 106006. [CrossRef]
Azizpanah, A.; Fathi, R.; Taki, M. Eco-energy and environmental evaluation of cantaloupe production by life cycle assessment
method. Environ. Sci. Pollut. Res. 2022, 30, 1854–1870. [CrossRef] [PubMed]
Taki, M.; Ajabshirchi, Y.; Ranjbar, S.F.; Rohani, A.; Matloobi, M. Modeling and experimental validation of heat transfer and energy
consumption in an innovative greenhouse structure. Inf. Process. Agric. 2016, 3, 157–174. [CrossRef]
Daliran, A.; Taki, M.; Marzban, A.; Rahnama, M.; Farhadi, R. Experimental evaluation and modeling the mass and temperature
of dried mint in greenhouse solar dryer; Application of machine learning method. Case Stud. Therm. Eng. 2023, 47, 103048.
[CrossRef]
Parajuli, S.; Narayan, T.; Gorjian, S.; Vithanage, M.; Raj Paudel, S. Assessment of potential renewable energy alternatives for a
typical greenhouse aquaponics in Himalayan Region of Nepal. Appl. Energy 2023, 344, 121270. [CrossRef]
Taki, M.; Ajabshirchi, Y.; Ranjbar, S.F.; Rohani, A.; Matloobi, M. Heat transfer and MLP neural network models to predict indoor
environment variables and energy lost in a semi-solar greenhouse. Energy Build. 2016, 110, 314–329. [CrossRef]
Figueiroa, V.; Torres, J.P.N. Simulation of a Small Smart Greenhouse. Designs 2022, 6, 106. [CrossRef]
Cafuta, D.; Dodig, I.; Cesar, I.; Kramberger, T. Developing a Modern Greenhouse Scientiﬁc Research Facility—A Case Study.
Sensors 2021, 21, 2575. [CrossRef]
Singh, R.K.; Rahmani, M.H.; Weyn, M.; Berkvens, R. Joint Communication and Sensing: A Proof of Concept and Datasets for
Greenhouse Monitoring Using LoRaWAN. Sensors 2022, 22, 1326. [CrossRef]

10. Rasheed, A.; Kwak, C.S.; Kim, H.T.; Lee, H.W. Building Energy an Simulation Model for Analyzing Energy Saving Options of

Multi-Span Greenhouses. Appl. Sci. 2020, 10, 6884. [CrossRef]

11. Ahamed, M.S.; Guo, H.; Tanino, K. Energy saving techniques for reducing the heating cost of conventional greenhouses. Biosyst.

Eng. 2019, 178, 9–23. [CrossRef]

12. Cao, Q.; Wu, Y.; Yang, J.; Yin, J. Greenhouse Temperature Prediction Based on Time-Series Features and LightGBM. Appl. Sci.

2023, 13, 1610. [CrossRef]

13. Petrakis, T.; Kavga, A.; Thomopoulos, V.; Argiriou, A.A. Neural Network Model for Greenhouse Microclimate Predictions.

Agriculture 2022, 12, 780. [CrossRef]

14. Zhang, C.; Liu, H.; Wang, C.; Zong, Z.; Wang, H.; Zhao, X.; Wang, S.; Li, Y. Testing and Analysis on the Spatial and Temporal

Distribution of Light Intensity and CO2 Concentration in Solar Greenhouse. Sustainability 2023, 15, 7001. [CrossRef]

15. Aamir, M.; Bhatti, M.A.; Bazai, S.U.; Marjan, S.; Mirza, A.M.; Wahid, A.; Hasnain, A.; Bhatti, U.A. Predicting the Environmental
Change of Carbon Emission Patterns in South Asia: A Deep Learning Approach Using BiLSTM. Atmosphere 2022, 13, 2011.
[CrossRef]
Faniyi, B.; Luo, Z.A. Physics-Based Modelling and Control of Greenhouse System Air Temperature Aided by IoT Technology.
Energies 2023, 16, 2708. [CrossRef]

16.

17. Chen, T.-H.; Lee, M.-H.; Hsia, I.-W.; Hsu, C.-H.; Yao, M.-H.; Chang, F.-J. Develop a Smart Microclimate Control System for

Greenhouses through System Dynamics and Machine Learning Techniques. Water 2022, 14, 3941. [CrossRef]

18. Bazgaou, A.; Fatnassi, H.; Bouharroud, R.; Tiskatine, R.; Wifaya, A.; Demrati, H.; Bammou, L.; Aharoune, A.; Bouirden, L. CFD
Modeling of the Microclimate in a Greenhouse Using a Rock Bed Thermal Storage Heating System. Horticulturae 2023, 9, 183.
[CrossRef]

19. Taki, M.; Abdanan Mehdizadeh, S.; Rohani, A.; Rahnama, M.; Rahmati-Joneidabad, M. Applied machine learning in greenhouse

simulation; new application and analysis. Inf. Process. Agric. 2018, 5, 253–268. [CrossRef]

20. Codeluppi, G.; Davoli, L.; Ferrari, G. Forecasting Air Temperature on Edge Devices with Embedded AI. Sensors 2021, 21, 3973.

[CrossRef]

Horticulturae 2023, 9, 853

18 of 18

21. Taki, M.; Rohani, A.; Sohelifard, F.; Abdeshahi, A. Assessment of energy consumption and modeling of output energy for wheat
production by neural network (MLP and RBF) and Gaussian process regression (GPR) models. J. Clean. Prod. 2018, 172, 3028–3041.
[CrossRef]

22. Rohani, A.; Taki, M.; Abdollahpour, Z. A novel soft computing model (Gaussian process regression with K-fold cross validation)

23.

for daily and monthly solar radiation forecasting (Part: I). Renew. Energy 2018, 115, 411–422. [CrossRef]
Jung, D.-H.; Lee, T.S.; Kim, K.; Park, S.H. A Deep Learning Model to Predict Evapotranspiration and Relative Humidity for
Moisture Control in Tomato Greenhouses. Agronomy 2022, 12, 2169. [CrossRef]

24. González-Vidal, A.; Mendoza-Bernal, J.; Ramallo, A.P.; Zamora, M.Á.; Martínez, V.; Skarmeta, A.F. Smart Operation of Climatic

Systems in a Greenhouse. Agriculture 2022, 12, 1729. [CrossRef]

25. Liu, R.; Yuan, S.; Han, L. Evaluation and Analysis on the Temperature Prediction Model for Bailing Mushroom in Jizhou, Tianjin.

Agriculture 2022, 12, 2044. [CrossRef]

26. Bolandnazar, E.; Rohani, A.; Taki, M. Energy consumption forecasting in agriculture by artiﬁcial intelligence and mathematical

27.

models. Energy Sources Part A Recover. Util. Environ. Eff. 2020, 42, 1618–1632. [CrossRef]
Jin, X.-B.; Zheng, W.-Z.; Kong, J.-L.; Wang, X.-Y.; Zuo, M.; Zhang, Q.-C.; Lin, S. Deep-Learning Temporal Predictor via Bidirectional
Self-Attentive Encoder–Decoder Framework for IOT-Based Environmental Sensing in Intelligent Greenhouse. Agriculture 2021,
11, 802. [CrossRef]

28. Ojo, M.O.; Zahid, A. Deep Learning in Controlled Environment Agriculture: A Review of Recent Advancements, Challenges and

Prospects. Sensors 2022, 22, 7965. [CrossRef]

29. Escamilla-García, A.; Soto-Zarazúa, G.M.; Toledano-Ayala, M.; Rivas-Araiza, E.; Gastélum-Barrios, A. Applications of Artiﬁcial
Neural Networks in Greenhouse Technology and Overview for Smart Agriculture Development. Appl. Sci. 2020, 10, 3835.
[CrossRef]
Jung, D.-H.; Kim, H.-J.; Kim, J.Y.; Lee, T.S.; Park, S.H. Model Predictive Control via Output Feedback Neural Network for
Improved Multi-Window Greenhouse Ventilation Control. Sensors 2020, 20, 1756. [CrossRef]

30.

31. Gong, L.; Yu, M.; Jiang, S.; Cutsuridis, V.; Pearson, S. Deep Learning Based Prediction on Greenhouse Crop Yield Combined TCN

and RNN. Sensors 2021, 21, 4537. [CrossRef]

32. Mahmood, F.; Govindan, R.; Bermak, A.; Yang, D.; Khadra, C.; Al-Ansari, T. Energy utilization assessment of a semi-closed

greenhouse using data-driven model predictive control. J. Clean. Prod. 2021, 324, 129172. [CrossRef]

33. Allouhi, A.; Choab, N.; Hamrani, A.; Saadeddine, S. Machine learning algorithms to assess the thermal behavior of a Moroccan

34.

agriculture greenhouse. Clean. Eng. Technol. 2021, 5, 100346. [CrossRef]
Francik, S.; Kurpaska, S. The Use of Artiﬁcial Neural Networks for Forecasting of Air Temperature indoor a Heated Foil Tunnel.
Sensors 2020, 20, 652. [CrossRef]

35. El Alaoui, M.; Chahidi, L.O.; Ugui, M.; Mechaqrane, A.; Allal, S. Evaluation of CFD and machine learning methods on predicting
greenhouse microclimate parameters with the assessment of seasonality impact on machine learning performance. Sci. Afr. 2023,
19, 01578. [CrossRef]

36. Mahmood, F.; Govindan, R.; Bermak, A.; Yang, D.; Al-Ansari, T. Data-driven robust model predictive control for greenhouse

temperature control and energy utilization assessment. Appl. Energy 2023, 343, 121190. [CrossRef]

37. Wang, S.; Gong, J.; Gao, H.; Liu, W.; Feng, Z. Gaussian Process Regression and Cooperation Search Algorithm for Forecasting

Nonstationary Runoff Time Series. Water 2023, 15, 2111. [CrossRef]

38. Ghosh, S.S.; Dey, S.; Bhogapurapu, N.; Homayouni, S.; Bhattacharya, A.; McNairn, H. Gaussian Process Regression Model for

Crop Biophysical Parameter Retrieval from Multi-Polarized C-Band SAR Data. Remote Sens. 2022, 14, 934. [CrossRef]

39. Taki, M.; Rohani, A. Machine learning models for prediction the Higher Heating Value (HHV) of Municipal Solid Waste (MSW)

for waste-to-energy evaluation. Case Stud. Therm. Eng. 2022, 31, 101823. [CrossRef]

40. Ali, A.; Hassanein, H.S. Wireless sensor network and deep learning for prediction greenhouse environments. In Proceedings
of the 2019 International Conference on Smart Applications, Communications and Networking (SmartNets), Sharm El Sheikh,
Egypt, 17–19 December 2019; pp. 1–5.

41. Castañeda-Miranda, A.; Castaño, V.M. Smart frost control in greenhouses by neural networks models. Comput. Electron. Agric.

2017, 137, 102–114. [CrossRef]

42. Yue, Y.; Quan, J.; Zhao, H.; Wang, H. The prediction of greenhouse temperature and humidity based on LM-RBF network. In
Proceedings of the 2018 IEEE International Conference on Mechatronics and Automation (ICMA), Changchun, China, 5–8 August
2018; pp. 1537–1541.

43. Panahi, F.; Najah Ahmed, A.; Singh, V.P.; Ehtearm, M.; Elshaﬁe, A.; Haghighi, A.T. Predicting freshwater production in seawater

greenhouses using hybrid artiﬁcial neural network models. J. Clean. Prod. 2021, 329, 129721. [CrossRef]

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual
author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to
people or property resulting from any ideas, methods, instructions or products referred to in the content.

