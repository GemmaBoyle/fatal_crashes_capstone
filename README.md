# Predicting Fatal Crash instances using the US Fars () database

## Overview

There were 33,244 fatal motor vehicle crashes in the United States in 2019 in which 36,096 deaths occurred. On average it costs $1200 to send each ambulance to a crash site. This project considers the factors which predict fatal instances of crashes so that emergency services may be distributed accordingly, decreasing the number of excess crash deaths while also minimising the cost of response vehicles.

## Libraries, technologies and tools

- Numpy
- Pandas
- SciPy - Stats 
- Matplotlib
- Seaborn
- Yellow Brick - DiscriminationThreshold
- SKlearn (SelectKBest / chi2 / train_test_split / mean_squared_error / cross_val_score / log_loss / StandardScalar / OrdinalEncoder / MinMaxScaler / LogisitcRegression / LogisticRegressionCV / KNeighborsClassifier / confusion_matrix / plot_confusion_matrix / plot_roc_curve / accuracy_score / precision_score / f1_score / classification_report / roc_auc_score / average_precision_score / plot_precision_recall_curve / DecisionTreeClassifier / AdaBoostClassifier / GradientBoostingClassifier / RandomForestClassifier / naive_bayes / CategoricalNB / MultinomialNB / ExtraTreesClassifier / BaggingClassifier / DecisionTreeClassifier / MLPClassifier / GridSearchCV / RandomizedSearchCV)

## Table of Contents

1. [Objectives](##Objectives)
2. [Background](##Background)
3. [Data Cleaning and Feature Engineering](##Data_Cleaning_and_Feature_Engineering)
4. [Exploratory Data Analysis](##Exploratory_Data_Analysis)
5. [Modelling](##Modelling)
6. [Evaluation](##Evaluation)
7. [Limitations](##Limitations)
8. [Conclusions and decision recommendations](##Conclusions_and_decision_recommendations)
9. [Further investigations](##Further_investigations)
10. [Key learnings](##Key_learnings)

## Objectives

* Predict (via classification) whether a person involved in a fatal crash will be a fatality above the baseline of 63%
* Investigate and highlight the most important factors of an accident, person and vehicle which contribute to fatalities in crashes in order to distribute emergency services effectively
* To consider the monetary impact of amending probability thresholds when predicting classification 

## Background

The FARS (fatal analysis reporting system) database collects information on every fatal crash across the United States. The data for this project considers crashes in 2019 comprising initially of 57882 persons, with each instance an amalgamation of three auxiliary data frames consisting of information on the person themselves, the vehicle they were travelling in and the overall accident. 

## Data Cleaning and Feature Engineering

The data itself was made up exclusively of categorical variables, which in itself presented some interesting problems for feature choice. As the data is only concerned with fatal crashes, single person crashes (such as a single person in a car colliding with a tree) were excluded from the data due to certainty of death, as this would give us an artificially increased prediction of fatality. Pedestrians involved in crashes were also removed, as these instances do not include most of the features the project is interested in such as vehicle type / licence status etc. 

| Variable            | Outcomes                                                                | 
| ------------------- |:-----------------------------------------------------------------------:|
| Blood Alcohol       | Not Tested, Tested and Positive, Tested and Negative                    |
| Ejected             | 0,1                                                                     |  
| Person Type         | Occupant, Driver                                                        |   
| Age                 | Range: 12.5 to 66.5                                                     |  
| Safety Measure      | 0,1                                                                     |  
| Vehicles in crash   | 1, 2, >2                                                                |  
| Day of Week         | Weekend, Weekday                                                        |  
| Time of Day         | Daytime, Nighttime                                                      |  
| Manner of Collision | Angle, head-on, sideswipe, not collision with motor vehicle in transport|  
| State               | 50 US States                                                            |  
| Vehicle Type        | Car, Light Truck, Motorcycle, Bus, Large Truck                          |  
| Impact Point        | Front, Left Side, Rear, Right Side, Non-Collision                       |  
| Licence Status      | Valid, Invalid                                                          |  
| Speeding            | 0,1                                                                     |  
| Rollover            | 0,1                                                                     |  
| Target: Fatality    | 0,1                                                                     |  

**Notes:**

**Binary variables:** All binary variables excluding fatality are coded as 0 being ‘No’ and 1 being ‘Yes’. The fatality variable is coded with 0 being ‘non-fatality’ and 1 being ‘fatality’.

**Age:** The age of the person within the original data is collected as a series of binary markers with each giving information about whether the person was in a specific age bracket. For the purposes of this project the age was imputed as the median value from the smallest inferred age range and used as a continuous variable. 

**Safety measure:** This is a join of several data columns and represents whether the person was using an appropriate safety device, i.e if the person was on a motorcycle whether they were wearing a helmet, if the person was in another type of vehicle whether they were using a seatbelt. 

**Licence status:** This variable indicates whether the person driving the vehicle held a valid licence for that vehicle (a valid car licence for a car or a valid motorcyle licence for a motorcyle).

## Exploratory Data Analysis

EDA was used to highlight trends across the data and give an indication of interesting features for further investigation. 

![count vehicle type](##https://github.com/GemmaBoyle/fatal_crashes_capstone/blob/main/Images/Count_vehicle_type.png)

The graph above gives the first indication that motorcycle riders are at the highest risk of fatality, as this is the only vehicle where fatalities outnumber non-fatalities. Safer vehicles appear to be larger, i.e trucks and buses. 

![blood alcohol pie](##https://github.com/GemmaBoyle/fatal_crashes_capstone/blob/main/Images/blood_alcohol_pie.png)

As we can observe from the pie chart above only around 37% of those involved in a fatal car crash are given a blood alcohol test. This was the first instance in which the question of ‘when’ a blood alcohol test will be given by a police officer or paramedic, as the majority of those in the crashes are not given tests and of those that are the majority are negative. This question is addressed later in the project. 

![boxplot age safety measure](##https://github.com/GemmaBoyle/fatal_crashes_capstone/blob/main/Images/Boxplot_age_safety_measure.png)

The box plot above shows a similar distribution across the age ranges for using an appropriate safety measure (helmet or seatbelt), with the median value being slightly lower in the negative class, indicating that on average younger people are less likely to use a safety measure. Very concerning are the lower values on the negative class, indicating there were young children (around 12 years old) who were not wearing a seatbelt whilst being a passenger. 

![age distribution](##https://github.com/GemmaBoyle/fatal_crashes_capstone/blob/main/Images/Age_distribution.png)

The KDE histogram on the left shows the distribution of ages across all instances, we can see two significant peaks, one around 20 years old and one around 62 years old, showing that a large proportion of these crashes involve young people and older people. When we compare this against the distribution of fatalities we can see that a lower proportion of younger drivers are killed in these crashes than are involved, however a larger proportion of older drivers are killed in these crashes than are involved. This may indicate that age is a factor in determining fatality, with older people being more likely to be fatalities. 

**Feature Selection**

![kmeans feature selection](##https://github.com/GemmaBoyle/fatal_crashes_capstone/blob/main/Images/k_best_feature_importance.png)

At this point a Chi-squared test for association was run for each feature against the target variable of fatality in order to decrease the number of features in the set to prevent overfitting of models. Every feature in the data set showed significant association with the target except for ‘Time of day’ and ‘Day of week’, hence these variables were removed for the remainder of the project. The largest bar on the very left of the diagram represents the age variable. Using an arbitrary cut off point the top 20 most associated features were chosen, these are shown below;

## Modelling 

## Evaluation

## Limitations

## Conclusions and decision recommendations

## Further investigations

## Key learnings




