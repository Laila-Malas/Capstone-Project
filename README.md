# Flight Delay Prediction using Machine Learning
## Aim of Project: 
The incredible rapid change in the policies procedures and aviation during the COVID-19 caused a huge effect on this industry, IATA faced challenges in adapting an agile approach to accommodate during this pandemic, to have a better and fast decision making as to support the ecosystem as a start, the airlines as a core player and serving the travelers as end users. This report covers the model as a Proof of concept for the IATA to cover the US airlines as it is considered on a lower scale of the airlines in the world. Data analysis and Data mining and its techniques were addressed to study the data set of January- Feb 2020 to predict the cancelation and delayed flights in future. Classifications techniques and measures were addressed among the four chosen algorithms. Data visualization using Tableau used for data representation and key answers for the predefined KPIS. In order to help in the IATA upper management for better decision making and policies and rules updates.

![overview-of-aviation-employment](https://user-images.githubusercontent.com/65343600/176993242-3a95f62a-f111-4d39-b65a-217bf0d8268e.jpg)

## Dataset Description 
### Dataset Source: https://www.kaggle.com/datasets/akulbahl/covid19-airline-flight-delays-and-cancellations
This dataset covers COVID-19 flight cancellations and delays. All these data were aggregated from the department of transportation in the United State of America covering 1st January until 29 February 2020 and added as a version on the Kaggle website. This dataset contains 47 attributes that talk about the origin, the destination of the trip, the airline, etc. with 1048575 records.  All these attributes and records were analyzed to determine which flight will be delayed or not since this target contains either 0 or 1 (delayed or not delayed), these attributes were classified into two types:

1- Categorical Attributes such as FL_Date, MKT_UNIQUE_CARRIER, TAIL_NUM, ORIGION, and DEST 

2- Numerical Attributes such as Year, Month, ARR_Time, ARR_DEL15 and Taxi out.
