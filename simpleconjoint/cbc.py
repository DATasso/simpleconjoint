import itertools
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import xlsxwriter


def count(
    df: pd.DataFrame,
    col_chosen: str = "Chosen",
    attributes: List[str] = [],
    attribute_combinations: List[List[str]] = [],
    to_excel: bool = False,
) -> Tuple:
    """
    A function to perform a basic count analysis on the dataset.

    Parameters
    ----------
    df: pandas.DataFrame
        A pandas dataframe with the conjoint info for the analysis.
        Each attribute level is expected to have the attribute and a underscore as a prefix,
            e.g: Color_Blue (or the attribute itself, Price for example).
        The level column should contain the number 1 if the attribute level was part of the alternative.
        It also expect a column to check if the alternative was chosen or not ('Chosen' by default, 1 if it was chosen, 0 if not).

    attributes: list()
        Array of strs with the attributes wanted for the count analysis
        e.g: ['Color', 'Size', 'Brand']

    col_chosen: str
        String to use as chosen column name, e.g: "Selected" or "Chosen"

    attribute_combinations: list()
        Array of arrays with the combinations desired or the count analysis.
        e.g: [['Color', 'Size']]
            This will add to the results every possible combination between color levels and size levels (if the attributes are present)
            -> Color_Blue x Size_24, Color_Blue x Size_27, Color_Green x Size_24, etc...

    to_excel: bool
        A boolean to check if you want to save the results into a single excel file named Count_ + timestamp,
        each attribute and attribute combination is saved into a separate sheet.

    Returns
    ----------
    As a result, it returns a tuple with two dictionaries containing the dataframes for each result.

    dataframes_by_attribute: dict()
        Dictionary with attribute as keys and dataframe result as values.
        The keys come from the parameter 'attributes'

    dataframes_by_combination: dict()
        Dictionary with attribute combination repr() as keys and dataframe result as values.
        The keys come from the parameter 'attribute_combinations'
        e.g: If attribute_combinations = [["Color", "Size"]] -> one of the keys would be "['Color', 'Size']"

    """
    attribute_by_column = {}
    columns_by_attribute = defaultdict(list)
    valid_columns = []
    for col in df.columns:
        attribute = col.split("_", 1)[0]
        if attribute in attributes:
            columns_by_attribute[attribute].append(col)
            valid_columns.append(col)
            attribute_by_column[col] = attribute

    dataframes_by_attribute = {}
    for attribute in columns_by_attribute:
        dataframes_by_attribute[attribute] = pd.DataFrame()

    missing_attributes = set(attributes) - set(columns_by_attribute.keys())
    if missing_attributes:
        warnings.warn(
            f"Warning... The following attributes were not found {missing_attributes}, will continue with the ones that were found"
        )

    for col in valid_columns:
        new_dataframe = pd.DataFrame()
        times_chosen = df[(df[col] == 1) & (df[col_chosen] == 1)][col].count()
        times_not_chosen = df[(df[col] == 1) & (df[col_chosen] == 0)][col].count()
        total = df[df[col] == 1].shape[0]
        result = times_chosen / total
        attribute = attribute_by_column[col]
        col_summary = {
            "Attribute Level": col,
            "Times Chosen": times_chosen,
            "Times Not Chosen": times_not_chosen,
            "Total Appearances": total,
            "Count Result": result,
        }
        dataframes_by_attribute[attribute] = dataframes_by_attribute[attribute].append(
            col_summary, ignore_index=True
        )

    dataframes_by_combination = {}
    for attribute_combination in attribute_combinations:
        dataframes_by_combination[repr(attribute_combination)] = pd.DataFrame()
        combination_cols = []
        for attribute in attribute_combination:
            try:
                combination_cols.append(columns_by_attribute[attribute])
            except KeyError:
                raise Exception(f"Attribute {attribute} does not exist.")

        cartesian_product = itertools.product(*combination_cols)
        for product in cartesian_product:
            combination = list(product)
            times_chosen = df[
                (df[combination] == 1).all(1) & (df[col_chosen] == 1)
            ].shape[0]
            times_not_chosen = df[
                (df[combination] == 1).all(1) & (df[col_chosen] == 0)
            ][col].shape[0]
            total = df[combination][(df[combination] == 1).all(1)].shape[0]
            result = times_chosen / total
            combination_summary = {}
            for idx, full_name in enumerate(combination):
                attribute = attribute_combination[idx]
                try:
                    covariate = full_name.split("_")[1]
                except IndexError:
                    covariate = full_name
                combination_summary[attribute] = covariate
            combination_summary.update(
                {
                    "Times Chosen": times_chosen,
                    "Times Not Chosen": times_not_chosen,
                    "Total Appearances": total,
                    "Count Result": result,
                }
            )
            dataframes_by_combination[
                repr(attribute_combination)
            ] = dataframes_by_combination[repr(attribute_combination)].append(
                combination_summary, ignore_index=True
            )

    if to_excel:
        writer = pd.ExcelWriter(
            "Count_{}.xlsx".format(datetime.now().strftime("%y%m%d-%H%M%S"))
        )
        for attribute in dataframes_by_attribute:
            attribute_dataframe = dataframes_by_attribute[attribute]
            attribute_dataframe.to_excel(writer, sheet_name=f"{attribute}", index=False)

        for attribute_combination in attribute_combinations:
            combination_dataframe = dataframes_by_combination[
                repr(attribute_combination)
            ]
            excel_sheet_name = " x ".join(attribute_combination)
            combination_dataframe.to_excel(
                writer, sheet_name=excel_sheet_name, index=False
            )

        writer.save()
        writer.close()

    return dataframes_by_attribute, dataframes_by_combination


class HMNL_Result:
    """
    Object to get the hmnl results.
    A function to perform a basic count analysis on the dataset.

    Init Parameters
    ----------
    stan_fit: pystan.StanFit4Model
        Instance containing the fitted results from pystan.
        Read pystan docs for further info on StanFit4Model methods

    attributes: list()
        Array of strs with the attributes.
        e.g: ['Color', 'Size', 'Brand']

    covariates: str
        Array of strs with the covariates (attribute levels or conjoint parameters)

    """

    _summary: Dict = None
    _individual_utilities: pd.DataFrame = None
    _individual_importances: pd.DataFrame = None

    def __init__(
        self,
        stan_fit: object,
        attributes: List[str],
        covariates: List[str],
    ):
        self.stan_fit = stan_fit
        self.attributes = attributes
        self.covariates = covariates

    @property
    def summary(self):
        """
        A function to get the summarized samples in all chains.

        Returns
        ----------
        pystan.StanFit4Model summary() method -> Summarize samples (compute mean, SD, quantiles) in all chains.

        """
        if self._summary is not None:
            return self._summary

        self._summary = self.stan_fit.summary()
        return self._summary
