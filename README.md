# 311 Service Calls
Predicting Service Completion Time for City of San Antonio 311 Customer Service Calls
>311 is the City of San Antonio (COSA)’s non-emergency service contact line. It connected Bexar county residents with specially- trained customer service representatives who can help with city requests such as: garbage/recycle information, potholes, stray animals, code enforcement concerns, graffiti on private property, or other city request. 

# Project Information
For this project I am looking at data from the COSA website, looking to predict the length of time it takes to close a given 311 request. My goal is to generate insight from the exploratory data analysis and model predictions that will allow end users such as city leaders and COSA managers close the gap and complete more 311 customer service times on or before their scheduled SLA date. 

## Project Goal
The goal for this project is to successfully predict the time it takes to close a 311 service request, measured by the target variable `days_to_close`. 

## Executive Summary
#### Inital Thoughts
I think that location with higher median incomes have better service reponse times and more instances of cases closing by their SLA date. Some initial questions I had of the data:

#### Initial Questions
1. Does a report's location affect whether it closes by its sla data?
2. What impacts a case's `days_to_close` more, the city council district a case originates from or the zip code for for the reported case?

#### Steps to Reproduce
1. ✅ Read this README file for project details
2. Ensure you have the latest version of python installed
3. Pull down the project modules and source csv and excel files from this repository to import into your local directory files

## Data Dictionary
Target | Dtype | Description
:--- | :--- | :---
`days_to_close` | datetime | the total time in days it took to close the case; feature engineered from `close_date` - `open_date`


Variable | Dtype |  Description
:--- | :--- | :---
`category` | object | top level 311 service request category
`case_open` | datetime | the date a case was submitted
`sla_due` | dtype | each service request `category` has a due date assigned to the request, based on the dept division `dept_div`
`dept` | object | the City deaprtment to whom the case is assigned
`dept_div` | object | the department division within the City deaprtment to whom the case is assigned
`council_distr` | object | The Council District number from where the issue was reported
`zip` | object | the zip code for the reported case/service requested
`population` | int | the population for the zip code for the reported case/service requested
`avg_inc` | float | the avergae income for the zip code for the reported case/service requested
`lat` | float | the latitude coordinate for the zip code for the reported case/service requested
`long` | float | the longitude coordinate for the zip code for the reported case/service requested




