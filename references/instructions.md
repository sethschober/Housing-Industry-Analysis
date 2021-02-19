# Phase 2 Final Project

Another phase down--you're almost halfway there!

![awesome gif](figures/halfway-there.gif)

This will be your first of several modeling projects.  In this case, your goal is to build a model that provides inferential insights into real-world housing data.

## BACKGROUND

The factors that influence housing prices interest many people, from homeowners hoping to improve the market potential of their current homes, to policymakers making decisions about investment in public projects. Often we rely on the "expert judgement" of real estate professionals to determine which factors are the most important, but these recommendations may be out-of-date, irrelevant to our particular housing market, or otherwise inaccurate.

## PROJECT GOAL

A client in King County, WA wants to advise homeowners on **home improvement projects** that will **add to the sale value of their homes**.

This advice should be based on data from the most recent full calendar year, 2019.

These three claims can be addressed directly with the three datasets (from the King County Department of Assessments) described later in this document:

1. Enclosing a porch will increase the sale price of a home.<sup>1</sup>
2. Converting a garage to a bedroom is a good way to increase the sale price of a home.<sup>2</sup>
3. Upgrading to a forced-air heating system will increase the sale price of a home.<sup>3,4</sup>

**Your task is to build a linear regression model** to represent home sale prices in King County, and use it to advise homeowners on which home improvement projects will add to their home sale values.

## THE DATASET

For this project, you'll be working with the King County House Sales dataset. As with most real world data sets, the column names are not perfectly described, so you'll have to do some research or use your best judgment if you have questions relating to what the data means.

The data itself (CSV files) is located in the `data/raw/` directory, and the [data dictionary](./King_County_Home_Sales_Data_Dictionary.pdf) is located in the `references/` directory.

Additional information about the MAJOR and MINOR attributes can be found [here](https://www5.kingcounty.gov/sdc/Metadata.aspx?Layer=parcel#AttributeInfo).

The particular tables required for this analysis are:

 - Real Property Sales (`EXTR_RPSale.csv`)
 - Residential Building (`EXTR_ResBldg.csv`)
 - Parcel (`EXTR_Parcel.csv`)

There is also a table called "Look Up" that explains the meaning of many of the attributes in the above tables (e.g. the `Street Surface` attribute of a parcel is a number with lookup code 60, which indicates that `1` means `PAVED`, `2` means `GRAVEL`, `3` means `DIRT`, and `4` means `UNDEVELOPED`).  This table is located at `EXTR_LookUp.csv`.

## PROJECT REQUIREMENTS

At minimum, you should:

 - **Build a linear regression model with a target variable of home sale price, which is statistically valid such that any interpretation of coefficients are valid**
    - Do your best to have features meet the assumptions of a linear regression (no multicollinearity, linear with respect to the outcome variable, errors are normally distributed, etc.). Note: This is difficult! We will be looking most of all for _improvement_ on this score. In order to demonstrate improvement, you should create a (not very good) model that will serve as a kind of _baseline_. Then you can compare future regression models that you build with that baseline.
    - Try to maximize R<sup>2</sup> _without breaking any assumptions_
 - **Interpret the results of your model**
    - Report the coefficients as well as the statistical significance of each features
    - It is perfectly acceptable to report a non-finding here (e.g. "we were unable to find evidence to support the claim that adding X is associated with a change in home values")
- **Make a recommendation of home improvement projects for King County homeowners**

In any extra time:

 - Engineer additional features to improve R<sup>2</sup> (without violating assumptions), either with the existing datasets or by acquiring additional datasets.
 - Utilize additional statistical techniques other than linear regression to address research questions related to the claims investigated.
 - Make use of bootstrapping to generate confidence intervals for your regression parameters and then compare with them with the intervals that statsmodels generates.

## DELIVERABLES

1. A public GitHub repository.
    - Make a fork of this project repository
2. An `environment.yml` file that contains all the necessary packages needed to recreate your `linreg-env` conda environment.
    - A study group will be held covering conda environments Monday, February 22. You can reference conda documentation ][here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to learn how to create and export the `linreg-env` conda environment.
3. A standalone `src/` directory that stores all relevant source code.
    - All functions have docstrings that act as [professional-quality documentation](http://google.github.io/styleguide/pyguide.html#381-docstrings). 
    - If applicable, [well documented](https://www.sqlstyle.guide/) SQL queries with appropriate single-line or multiline comments.
    - Quality linear regression model
       - Your model should avoid violating any of the assumptions of a linear regression
       - Whenever necessary, briefly explain in comments the changes made from one iteration to the next, and why you made these choices
4. A standalone `data/` directory that stores all relevant raw and processed data files
    - Raw data is already located in `data/raw/`
    - When you have a dataframe with all of your cleaned data, consider saving it in the `data/processed` directory
5. A standalone `references/` directory that stores all relevant literature, data dictionaries, or useful references that were used to help you during the project.
6. A standalone `reports/` directory that stores your `presentation.pdf` file
7. A standalone `notebooks/` directory that stores both your exploratory and report notebooks
    - A record of your workflow should be stored in `notebooks/exploratory`. Don't be afraid to leave in error messages, so you know what didn't work!
8. A user-focused `README.md` file that briefly covers your process, methodology and findings.
    - Someone with no context on your project should be able to use this document to understand the structure of your project, and adapt your code for their needs.
9. One final Jupyter Notebook file stored in `notebooks/report` that focuses on visualization and presentation
    - The very beginning of the notebook contains a description of the purpose of the notebook.
       - This is helpful for your future self and anyone of your colleagues that needs to view your notebook. Without this context, you’re implicitly asking your peers to invest a lot of energy to help solve your problem. Help them to jump into your project by providing them the purpose of this Jupyter Notebook.
    - Explanation of the data sources and where one can retrieve them
       - Whenever possible, link to the corresponding data dictionary
    - Custom functions are imported from Python modules and are not created directly in the notebook. As soon as you have a working function in one of your exploratory notebooks, copy it over to `src` so it is reusable.
    - At least 3 meaningful data visualizations, with corresponding interpretations. All visualizations are well labeled with axes labels, a title, and a legend (when appropriate)
    - Take the time to make sure that you craft your story well, and clearly explain your process and findings in a way that clearly shows both your technical expertise and your ability to communicate your results!
10. An "Executive Summary" Keynote/PowerPoint/Google Slide presentation (delivered as a PDF export) that explains what you have found.
    - Make sure to also add and commit this file as presentation.pdf of your non-technical presentation to your repository with a file name of `reports/presentation.pdf`.
    - Contain between 5-10 professional quality slides detailing:
       - A high-level overview of your methodology
       - The results you’ve uncovered
       - Any real-world recommendations you would like to make based on your findings (ask yourself--why should the executive team care about what you found? How can your findings help the company/stakeholder?)
       - Avoid technical jargon and explain results in a clear, actionable way for non-technical audiences.
    - The slides should use visualizations whenever possible, and avoid walls of text
    - All visualizations included in this presentation should also be exported as image files (e.g. with `plt.savefig`, not by taking a screenshot) and saved under `reports/figures/`


## THE PROCESS

These steps are informed by Smart Vision's<sup>9</sup> description of the CRISP-DM process.

### 1. Business Understanding

Start by reading this document, and making sure that you understand the kinds of questions being asked.

Three things to be sure you establish during this phase are:

1. **Objectives:** what questions are you trying to answer, and for whom?
2. **Project plan:** what are you going to accomplish on each day until the project deadline?
3. **Success criteria:** what does a successful project look like? How will you know when you have achieved it?

### 2. Data Understanding

Read through the data dictionary, and perform some initial exploratory data analysis (EDA).

**Hint:** A lot of the data in these CSVs looks numeric, but actually represents some kind of ID.  So we recommend opening the data in Pandas with something like:

```python
pd.read_csv("../../data/raw/EXTR_Parcel.csv", dtype=str)
```

You can always convert columns that are actually numbers into numbers later, but this technique means that strings like `"0410"` aren't converted into `410` accidentally.

It may be useful to generate visualizations of the data during this phase.

### 3. Data Preparation

Through SQL and/or Pandas, perform any necessary data cleaning and develop a query that pulls in all relevant data for analysis in a linear regression model, including any merging of tables. Be sure to document any data that you choose to drop or otherwise exclude. This is also the phase to consider any feature scaling or one-hot encoding required to feed the data into a linear regression model.

### 4. Modeling

The modeling phase in this project should be a brief stop-over as you are jumping back and forth between the data preparation and the evaluation phases.  If the data preparation was done correctly, it only takes a few lines of code to build a linear regression model.  Then you should be able to print out your model's metrics and quickly move to the evaluation phase.

(In future modules, there will be more complex steps involved with tuning the model itself.)

### 5. Evaluation

Your goal here is not to build just one model; it is to build many models and then choose the best one. In this case (which is different from future machine learning models), your goal is to gain additional insight into the housing sales data from 2019 in King County, as opposed to building a generalizable model that can make predictions about unseen data.

**In other words, you are not trying to build a tool to compete with the Zillow Zestimate to predict home values.**

Because of this framing, a higher R<sup>2</sup> is ideal, but you should not focus on attaining a higher R<sup>2</sup> so long as you have gross violations of the assumptions of a linear regression. These assumptions should be your focus both when selecting features and when assessing the quality of your model. Even with a relatively low R<sup>2</sup>, coefficients and p-values will be valid and meaningful if there are no violations of the assumptions.

This framing has some implications for your process. For example, you do not want to perform a train-test split or have some other form of hold-out data; you want all of your data to go into getting the best possible coefficients. There is also no single number you can reference to demonstrate that you have the best possible model; in this way causal inference is more of an art, whereas machine learning is more of a science. You can use as many features as you want, but every step of the way you need to check the assumptions, not just check whether R<sup>2</sup> has improved. Because R<sup>2</sup> can only go up when you add more features, you may want to consult the adjusted R<sup>2</sup> metric as well.

Your first cycle through these steps should result in some kind of "baseline" model.  Once you have that model, create a mini "report" that describes the model performance (in terms of R<sup>2</sup> or adjusted R<sup>2</sup>) as well as checks for the assumptions of linear regression.  The Learn labs [Assumptions for Linear Regression](https://learning.flatironschool.com/courses/1385/pages/assumptions-for-linear-regression?module_item_id=91911), [Ordinary Least Squares in Statsmodels (OLS)](https://learning.flatironschool.com/courses/1385/assignments/47942?module_item_id=91912), and [Regression Diagnostics in Statsmodels](https://learning.flatironschool.com/courses/1385/assignments/47960?module_item_id=91915) are a good starting point for this, but you are also free to use external resources, e.g. looking through the statistics available in the [Statsmodels `stats` submodule](https://www.statsmodels.org/stable/stats.html).

As you build each new model, perform these checks repeatedly to see whether the new models are better than the baseline.  *You will almost certainly want to build functions to perform these checks* once you get into a rhythm of tweaking the model slightly, evaluating the result, rinse, repeat.

### 6. Deployment

When you are approaching the end of the available time, choose your best model and report what it says about your research questions. An example finding you might report is: "There is a statistically significant relationship between `<x variable>` and housing price. For every increase of 1 `<x variable>`, the housing price increases by `<amount>`, all else being equal". Consider what types of visualizations would help to communicate the scale and direction of these findings.

Beyond just the numbers, tie these findings into a broader narrative that incorporates your business understanding.  Complete the deliverables listed above, and make sure you can answer the following questions about your process:

 - "Why are these questions important from a business perspective?"
 - "How did you decide on the data cleaning options you performed?"
 - "Why did you choose a given method or library?"
 - "Why did you select those visualizations and what did you learn from each of them?"
 - "Why did you pick those features as predictors?"
 - "How would you interpret the results?"
 - "How confident are you in the predictive quality of the results?"
 - "What are some of the things that could cause the results to be wrong?"

## Citations
1. Unknown author. 2017. "What is the Value of Adding a Porch Enclosure to my Outdoor Space?". Patio Enclosures. Available at https://www.patioenclosures.com/what-is-the-value-of-adding-a-porch-enclosure-to-my-outdoor-space.aspx 
2. Backman M. 2019. "Should You Convert Your Garage to Extra Living Space?". millionacres. Available at: https://www.fool.com/millionacres/real-estate-market/home-renovations/should-you-convert-your-garage-extra-living-space/
3. Unknown author. 2019. "How Your HVAC System Affects Home Value". Big Mountain Air. Available at: https://www.bigmountainair.com/hvac-system-home-value/#:~:text=The%20impact%20of%20HVAC%20on,is%20a%20valuable%20selling%20point
4. "Can A New HVAC Unit Increase The Overall Value Of Your Home?". Crystal Heating and Cooling. Available at: https://crystalheatingandcooling.com/new-hvac-unit-increase-the-overall-value-of-your-home/



