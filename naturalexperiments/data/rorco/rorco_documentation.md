# RORCO Dataset Creation: Detailed Documentation

---

## Table of Contents

- [RORCO Dataset Creation: Detailed Documentation](#rorco-dataset-creation-detailed-documentation)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Data Sources](#data-sources)
  - [Environment Setup](#environment-setup)
  - [Core Functions](#core-functions)
    - [1. Geocoding Addresses](#1-geocoding-addresses)
      - [a. `get_response(address)`](#a-get_responseaddress)
      - [b. `extract_colorado(response)`](#b-extract_coloradoresponse)
      - [c. `add_geometry(data, dataname)`](#c-add_geometrydata-dataname)
    - [2. Extracting School and District Numbers](#2-extracting-school-and-district-numbers)
      - [a. `extract_school_number(school)`](#a-extract_school_numberschool)
      - [b. `extract_district_number(school)`](#b-extract_district_numberschool)
    - [3. Merging School Data](#3-merging-school-data)
      - [`add_school_data(school_geom)`](#add_school_dataschool_geom)
    - [4. Loading and Processing School Assessment Data](#4-loading-and-processing-school-assessment-data)
      - [`load_school_data()`](#load_school_data)
    - [5. Loading and Processing Clinic Data](#5-loading-and-processing-clinic-data)
      - [`load_clinic_data()`](#load_clinic_data)
    - [6. Assigning Clinic Visits to Schools](#6-assigning-clinic-visits-to-schools)
      - [`assign_visits_to_schools(school_geom, clinics_geom)`](#assign_visits_to_schoolsschool_geom-clinics_geom)
    - [7. Adding Proximity and Rural Indicators](#7-adding-proximity-and-rural-indicators)
      - [`add_proximity(school_geom)`](#add_proximityschool_geom)
  - [Dataset Generation Workflow](#dataset-generation-workflow)
  - [Final Output](#final-output)
  - [Appendix](#appendix)
    - [A. List of Acronyms](#a-list-of-acronyms)

---

## Introduction

The RORCO (Reach Out and Read Colorado) dataset aims to integrate various educational and health-related data sources to analyze the impact of early childhood interventions on academic performance. This documentation provides a comprehensive overview of the process undertaken to create the RORCO dataset, detailing each step, data source, and processing logic implemented in the provided code.

---

## Data Sources

The dataset creation process leverages multiple data sources, primarily focused on schools in Colorado and health clinic data. Below is a summary of the primary data sources used:

1. **Google Geocoding API**
   - **Purpose**: To obtain geographical coordinates (latitude and longitude) for physical addresses.
   - **Usage**: Geocoding school and clinic addresses.

2. **Colorado Department of Education (CDE) Datasets**
   - **Free and Reduced Lunch Data**
     - **File**: `2022-23_K12_FRL_bySchool - Suppressed.xlsx`
     - **Purpose**: Provides information on the number of students eligible for free or reduced lunch, an indicator of socioeconomic status.
   - **IPST (Instructional Program Service Type) Data**
     - **File**: `2022-23_IPST_bySchool - Suppressed.xlsx`
     - **Purpose**: Details various instructional programs and services offered at schools.
   - **Demographic Data**
     - **File**: `2022-23_Membership_Grade_bySchool.xlsx`
     - **Purpose**: Contains student membership counts by grade level.
   - **Attendance Data**
     - **File**: `2019-20_Attendance and Truancy Rates by School.xlsx`
     - **Purpose**: Provides attendance and truancy rates, offering insights into student engagement.
   - **Student-Teacher Ratio Data**
     - **File**: `2022.2023PupilTeacherRatio.publishedreport.xlsx`
     - **Purpose**: Indicates the ratio of students to teachers, a measure often linked to educational quality.
   - **Average Teacher Salary Data**
     - **File**: `2022.2023AverageTeachersSalary.FTE_.PublishedReport.updated4.13.23.xlsx`
     - **Purpose**: Details average teacher salaries by district, reflecting potential resource allocations.

3. **School Assessment Data**
   - **Files**: Multiple files named in the pattern `State Assessment Data Lab_{year}_{suffix}.xlsx`, covering academic years from 2014-2015 to 2018-2019.
   - **Purpose**: Provides assessment scores and related data for various schools.

4. **School Addresses**
   - **File**: `Public School Mailing Labels 2023-2024.csv`
   - **Purpose**: Contains mailing and physical addresses of public schools in Colorado.

5. **Clinic Data**
   - **File**: `rorco_cumulative.csv`
   - **Purpose**: Contains data from clinics, including well-child visits and book distributions.

6. **School District Boundaries**
   - **File**: `CDPHE_CDOE_School_District_Boundaries.shp`
   - **Purpose**: Shapefile defining the geographical boundaries of school districts in Colorado.

---

## Environment Setup

The process relies on several Python libraries for data manipulation, geospatial analysis, and API interactions. Below is a summary of the primary libraries used:

- **Geopandas (`gpd`)**: For handling geospatial data.
- **Pandas (`pd`)**: For data manipulation and analysis.
- **Requests**: For making HTTP requests, particularly to the Google Geocoding API.
- **Shapely**: For geometric operations.
- **NumPy (`np`)**: For numerical operations.
- **Pickle**: For serializing and deserializing Python objects.
- **Difflib**: For comparing sequences, used here for matching similar strings.
- **Matplotlib**: For plotting (used in plotting functions).
- **SciPy**: For spatial queries.

**Note**: The Google Geocoding API requires an API key. Ensure you have one set up and replace the placeholder in the code.

---

## Core Functions

### 1. Geocoding Addresses

#### a. `get_response(address)`

- **Purpose**: Sends a geocoding request to the Google Geocoding API for a given address and retrieves the JSON response.
- **Parameters**:
  - `address` (str): The physical address to geocode.
- **Process**:
  - Formats the address by replacing spaces with '+' and removing '#' characters.
  - Constructs the API request URL using the formatted address and the Google API key.
  - Sends a GET request to the API and parses the JSON response.
- **Returns**: JSON response from the API.

#### b. `extract_colorado(response)`

- **Purpose**: Filters the API response to extract the result corresponding to Colorado addresses.
- **Parameters**:
  - `response` (dict): JSON response from the Google Geocoding API.
- **Process**:
  - Iterates through the results in the response.
  - For each result, checks the address components to identify if 'Colorado' is present.
  - Returns the first result matching 'Colorado'.
- **Returns**: The JSON object representing the Colorado-based address.

#### c. `add_geometry(data, dataname)`

- **Purpose**: Adds geographical coordinates (latitude and longitude) and geometric points to the dataset based on addresses.
- **Parameters**:
  - `data` (DataFrame): The dataset containing addresses.
  - `dataname` (str): A label for the dataset, used in naming the pickle file storing geocoded addresses.
- **Process**:
  - Extracts unique addresses from the 'Combined Address' column.
  - Checks if a pickle file (`address_to_location.dict`) exists to avoid redundant API calls.
  - For addresses not already geocoded, calls `get_response` and `extract_colorado` to obtain latitude and longitude.
  - Stores the mapping of addresses to their geographical locations in the pickle file for future use.
  - Constructs a GeoDataFrame with the addresses and their corresponding geometries.
  - Merges the original data with the geocoded data based on the 'Combined Address'.
- **Returns**: A GeoDataFrame with added geographical information.

### 2. Extracting School and District Numbers

#### a. `extract_school_number(school)`

- **Purpose**: Extracts the school code from a string containing the school's name and code.
- **Parameters**:
  - `school` (str): The string containing the school information, typically in the format 'School Name (Code)'.
- **Process**:
  - Identifies the position of the last opening parenthesis.
  - Extracts the substring within the parentheses, which represents the school code.
  - Converts the extracted code to an integer.
- **Returns**: Integer representing the school code.

#### b. `extract_district_number(school)`

- **Purpose**: Extracts the district code from a string containing the district's name and code.
- **Parameters**:
  - `school` (str): The string containing the district information, typically in the format 'District Name (Code)'.
- **Process**:
  - Removes any occurrences of '(J)' from the string, which might represent joint districts.
  - Identifies the substring within the parentheses, which represents the district code.
  - Converts the extracted code to an integer.
- **Returns**: Integer representing the district code.

### 3. Merging School Data

#### `add_school_data(school_geom)`

- **Purpose**: Integrates various datasets containing school-related information into the main GeoDataFrame.
- **Parameters**:
  - `school_geom` (GeoDataFrame): The GeoDataFrame containing school geometries and basic information.
- **Process**:
  - **Loading Supplementary Datasets**:
    - Free and Reduced Lunch Data (`school_frl`)
    - IPST Data (`school_ipst`)
    - Demographic Data (`school_membership`)
    - Attendance Data (`school_attendance`)
    - Student-Teacher Ratio Data (`school_ratio`)
    - Average Teacher Salary Data (`district_salary`)
  - **Data Cleaning**:
    - Replacing placeholder values (e.g., '*') with zeros in datasets where necessary.
    - Renaming columns for consistency.
  - **Merging Datasets**:
    - Sequentially merges each school-level dataset with `school_geom` based on the 'School Code'.
    - For district-level data (e.g., `district_salary`), first extracts the 'District Code' from the 'State/District/School' column and then merges based on 'District Code'.
    - Handles duplicate columns resulting from merges by renaming and dropping as necessary.
- **Returns**: Enhanced GeoDataFrame with merged school and district data.

### 4. Loading and Processing School Assessment Data

#### `load_school_data()`

- **Purpose**: Loads, processes, and integrates school assessment data spanning multiple academic years.
- **Process**:
  - **Data Loading**:
    - Iterates through the specified academic years and file suffixes to load assessment data from multiple Excel files.
    - Concatenates the loaded data into a single DataFrame.
  - **Data Cleaning**:
    - Replaces empty strings with NaN and forward-fills missing values in key columns.
    - Filters out rows where 'Mean Scale Score' is missing or marked with '-'.
  - **Extracting Identifiers**:
    - Uses `extract_school_number` to obtain 'School Code' from the 'State/District/School' column.
  - **Address Integration**:
    - Loads school addresses from `Public School Mailing Labels 2023-2024.csv`.
    - Merges address data with the main DataFrame based on 'School Code'.
    - Constructs a 'Combined Address' for geocoding.
  - **Geocoding**:
    - Calls `add_geometry` to add geographical coordinates and geometries based on addresses.
  - **Merging Supplementary Data**:
    - Calls `add_school_data` to integrate additional school and district datasets.
  - **Indexing**:
    - Creates a unique 'triple_index' for each record based on the academic year, grade, and school code.
    - Sets 'triple_index' as the index of the GeoDataFrame.
- **Returns**: A comprehensive GeoDataFrame containing school assessment data enriched with geographical and supplementary information.

### 5. Loading and Processing Clinic Data

#### `load_clinic_data()`

- **Purpose**: Loads, processes, and integrates clinic data, associating it with corresponding school district boundaries.
- **Process**:
  - **Loading Data**:
    - Reads school district boundaries from the shapefile `CDPHE_CDOE_School_District_Boundaries.shp`.
    - Loads clinic data from `rorco_cumulative.csv`.
  - **Data Cleaning**:
    - Converts columns related to book distributions and visits to numeric types.
  - **Address Integration**:
    - Constructs a 'Combined Address' for each clinic based on its physical location.
    - Calls `add_geometry` to add geographical coordinates and geometries based on clinic addresses.
  - **Spatial Join**:
    - Performs a spatial join between the clinic geometries and school district boundaries to associate each clinic with a school district.
    - Renames columns to reflect the district code and name.
- **Returns**: A GeoDataFrame containing clinic data with geographical and district information.

### 6. Assigning Clinic Visits to Schools

#### `assign_visits_to_schools(school_geom, clinics_geom)`

- **Purpose**: Distributes clinic visit data across nearby schools based on proximity and other criteria.
- **Parameters**:
  - `school_geom` (GeoDataFrame): GeoDataFrame containing school data.
  - `clinics_geom` (GeoDataFrame): GeoDataFrame containing clinic data.
- **Process**:
  - **Aggregating Clinic Visits**:
    - Groups clinic data by report submission year and geometry, summing the 'Well-Child Visits' for children aged 6 months to 5 years.
  - **Identifying Nearby Schools**:
    - For each clinic, identifies up to 100 nearby schools using spatial querying (`cKDTree`).
  - **Allocating Visits Based on Age Ratios**:
    - Defines age groups (1 to 6 years) and their corresponding ratios of total visits.
    - For each clinic and age group, calculates the number of children and distributes them across nearby schools.
  - **Mapping to School Grades and Years**:
    - Converts child ages to corresponding school grades based on the year of the clinic visit and the academic year.
    - Associates clinic visits with school grades (from 3rd to 9th).
  - **Assigning Students to Schools**:
    - For each school, ensures that the number of students from RORCO does not exceed 80% of the school's capacity.
    - Updates the 'num_from_RORCO' field to reflect the number of students associated with RORCO visits.
  - **Indicator for RORCO Schools**:
    - Adds an 'is_RORCO' indicator, set to 1 if more than 75% of a school's capacity is filled by RORCO-associated students.
- **Returns**: Updated `school_geom` GeoDataFrame with RORCO visit allocations.

### 7. Adding Proximity and Rural Indicators

#### `add_proximity(school_geom)`

- **Purpose**: Adds information about the number of nearby students and a rural indicator for each school.
- **Parameters**:
  - `school_geom` (GeoDataFrame): GeoDataFrame containing school data.
- **Process**:
  - **Coordinate Transformation**:
    - Converts the school's geometries to the 'EPSG:3857' coordinate reference system, which uses meters.
  - **Spatial Querying**:
    - For each school, identifies all other schools within a 10-mile radius (~16,093 meters) using spatial querying (`cKDTree`).
  - **Calculating Nearby Students**:
    - Sums the capacities of all schools within the 10-mile radius to determine the number of nearby students.
  - **Rural Indicator**:
    - Sets the 'is_rural' indicator to 1 for schools with fewer than 10,000 nearby students, based on the assumption that 20% of the Colorado population is children.
- **Returns**: Enhanced `school_geom` GeoDataFrame with proximity and rural indicators.

---

## Dataset Generation Workflow

The process of generating the RORCO dataset involves orchestrating the aforementioned functions in a logical sequence. Below is the step-by-step workflow:

1. **Loading and Processing School Data**:
   - **Function**: `load_school_data()`
   - **Outcome**: A GeoDataFrame (`school_geom`) containing school assessment data, addresses, and supplementary information.

2. **Loading and Processing Clinic Data**:
   - **Function**: `load_clinic_data()`
   - **Outcome**: A GeoDataFrame (`clinics_geom`) containing clinic data with geographical and district associations.

3. **Assigning Clinic Visits to Schools**:
   - **Function**: `assign_visits_to_schools(school_geom, clinics_geom)`
   - **Outcome**: Updated `school_geom` with allocations of clinic visits, indicating RORCO-associated students.

4. **Adding Proximity and Rural Indicators**:
   - **Function**: `add_proximity(school_geom)`
   - **Outcome**: Finalized `school_geom` with proximity metrics and rural indicators.

5. **Exporting the Dataset**:
   - **Process**:
     - Writes the finalized `school_geom` GeoDataFrame to a CSV file named `rorco_data.csv`.

**Note**: The main execution block orchestrates these steps, ensuring the dataset is generated and saved.

---

## Final Output

The culmination of the process is a CSV file named `rorco_data.csv`, which encapsulates a comprehensive dataset integrating school assessment data, clinic visit allocations, geographical information, and various indicators. Each record in the dataset represents a unique combination of academic year, grade, and school, enriched with metrics such as:

- Geographical Coordinates (Latitude, Longitude)
- School and District Identifiers
- Assessment Scores
- Socioeconomic Indicators (e.g., Free and Reduced Lunch Eligibility)
- Clinic Visit Allocations (e.g., Number of Students from RORCO)
- Proximity Metrics (e.g., Number of Nearby Students)
- Rural Indicator

This dataset serves as a valuable resource for analyzing the impact of early childhood interventions on academic outcomes within the state of Colorado.

---

## Appendix

### A. List of Acronyms

- **RORCO**: Reach Out and Read Colorado
- **CDE**: Colorado Department of Education
- **IPST**: Instructional Program Service Type
- **CDPHE**: Colorado Department of Public Health and Environment
- **CMAS**: Colorado Measures of Academic Success