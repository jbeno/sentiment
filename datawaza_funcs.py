# Standard library imports
import os
import time

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.metrics import (mean_absolute_error, mean_squared_error, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, precision_recall_curve, PrecisionRecallDisplay,
                             roc_auc_score, make_scorer, precision_score, recall_score, f1_score, accuracy_score)

# Weights and Biases
import wandb

# Typing imports
from typing import Optional, Union, Tuple, List, Dict, Any

class DebugPrinter:
    """
    Conditionally print debugging information during the execution of a script.

    This class provides a simple way to print debugging information during the
    execution of a script. By setting the `debug` attribute to True, you can
    enable or disable debugging output throughout the script. The `print()`
    method works like the built-in `print()` function but only prints output
    when debugging is enabled.

    Use this class when you need to easily control and print debugging messages
    in your script, allowing you to enable or disable debugging output as
    needed. It allows you to avoid nesting a bunch of print statements
    underneath an "if debug:" statement, and it's lighter weight than a full
    logging setup.

    Parameters
    ----------
    debug : bool, optional
        Whether to enable debugging output. Default is False.

    Examples
    --------
    Set some test variables for the examples:

    >>> name = 'Setting'
    >>> value = 10

    Example 1: Create a DebugPrinter object and print a debug message:

    >>> db = DebugPrinter(debug=True)
    >>> db.print('This is a debug message.')
    This is a debug message.

    Example 2: Disable debugging and print a message that doesn't display:

    >>> db.set_debug(False)
    >>> db.print("This is a debug message that won't show.")

    Example 3: Re-enable debug, and print a formatted message with variables:

    >>> db.set_debug(True)
    >>> db.print(f'This is a debug message. ({name}: {value})')
    This is a debug message. (Setting: 10)
    """

    def __init__(
            self,
            debug: bool = False
    ):
        """
        Initialize the DebugPrinter object with the specified debugging setting.
        """
        self.debug = debug

    def print(self, *args, **kwargs):
        """
        Print debugging information if debugging is enabled.

        Parameters
        ----------
        *args
            Any number of positional arguments to print.
        **kwargs
            Any keyword arguments to pass to the built-in `print()` function.
        """
        if self.debug:
            print(*args, **kwargs)

    def set_debug(self, debug: bool):
        """
        Set the debugging setting to enable or disable debugging output.

        Parameters
        ----------
        debug : bool
            Whether to enable or disable debugging output.
        """
        self.debug = debug


def eval_model(
        *,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        class_map: Dict[Any, Any] = None,
        estimator: Optional[Any] = None,
        x_test: Optional[np.ndarray] = None,
        class_type: Optional[str] = None,
        pos_label: Optional[Any] = 1,
        threshold: float = 0.5,
        multi_class: str = 'ovr',
        average: str = 'macro',
        title: Optional[str] = None,
        model_name: str = 'Model',
        class_weight: Optional[str] = None,
        decimal: int = 2,
        bins: int = 10,
        bin_strategy: str = None,
        plot: bool = False,
        save_plots: bool = False,
        save_dir: str = 'plots',
        figsize: Tuple[int, int] = (12, 11),
        figmulti: float = 1.7,
        conf_fontsize: int = 14,
        return_metrics: bool = False,
        output: bool = True,
        debug: bool = False,
        wandb_run: Optional[Any] = None
) -> Optional[Dict[str, Union[int, float]]]:
    """
    Evaluate a classification model's performance and plot results.

    This function provides a comprehensive evaluation of a binary or multi-class
    classification model based on `y_test` (the actual target values) and `y_pred`
    (the predicted target values). It displays a text-based classification report
    enhanced with True/False Positives/Negatives (if binary), and 4 charts if
    `plot` is True: Confusion Matrix, Histogram of Predicted Probabilities, ROC
    Curve, and Precision-Recall Curve.

    If `class_type` is 'binary', it will treat this as a binary classification.
    If `class_type` is 'multi', it will treat this as a multi-class problem. If
    `class_type` is not specified, it will be detected based on the number of
    unique values in `y_test`. To plot the curves or adjust the `threshold`
    (default 0.5), both `x_test` and `estimator` must be provided so that
    proababilities can be calculated.

    For binary classification, `pos_label` is required. This defaults to 1 as an
    integer, but can be set to any value that matches one of the values in
    `y_test` and `y_pred`. The `class_map` can be used to provide display names
    for the classes. If not provided, the actual class values will be used.

    A number of classification metrics are shown in the report: Accuracy,
    Precision, Recall, F1, and ROC AUC. In addition, for binary classification,
    True Positive Rate, False Positive Rate, True Negative Rate, and False
    Negative Rate are shown. The metrics are calculated at the default threshold
    of 0.5, but can be adjusted with the `threshold` parameter.

    You can customize the `title` of the report completely, or pass the
    `model_name` and it will be displayed in a dynamically generated title. You
    can also specify the number of `decimal` places to show, and size of the
    figure (`fig_size`). For multi-class, you can set a `figmulti` scaling factor
    for the plot.

    You can set the `class_weight` as a display only string that is not used in
    any functions within `eval_model`. This is useful if you trained the model
    with a 'balanced' class_weight, and now want to pass that to this report to
    see the effects.

    A dictionary of metrics can be returned if `return_metrics` is True, and
    the output can be disabled by setting `output` to False. These are used by
    parent functions (ex: `compare_models`) to gather the data into a DataFrame
    of the results.

    Use this function to assess the performance of a trained classification
    model. You can experiment with different thresholds to see how they affect
    metrics like Precision, Recall, False Positive Rate and False Negative
    Rate. The plots make it easy to see if you're getting good separation and
    maximum area under the curve.

    Parameters
    ----------
    y_test : np.ndarray
        The true labels of the test set.
    y_pred : np.ndarray
        The predicted labels of the test set.
    class_map : Dict[Any, Any], optional
        A dictionary mapping class labels to their string representations.
        Default is None.
    estimator : Any, optional
        The trained estimator object used for prediction. Required for
        generating probabilities. Default is None.
    x_test : np.ndarray, optional
        The test set features. Required for generating probabilities.
        Default is None.
    class_type : str, optional
        The type of classification problem. Can be 'binary' or 'multi'.
        If not provided, it will be inferred from the number of unique labels.
        Default is None.
    pos_label : Any, optional
        The positive class label for binary classification.
        Default is 1.
    threshold : float, optional
        The threshold for converting predicted probabilities to class labels.
        Default is 0.5.
    multi_class : str, optional
        The method for handling multi-class ROC AUC calculation.
        Can be 'ovr' (one-vs-rest) or 'ovo' (one-vs-one).
        Default is 'ovr'.
    average : str, optional
        The averaging method for multi-class classification metrics.
        Can be 'macro', 'micro', 'weighted', or 'samples'.
        Default is 'macro'.
    title : str, optional
        The title for the plots. Default is None.
    model_name : str, optional
        The name of the model for labeling the plots. Default is 'Model'.
    class_weight : str, optional
        The class weight settings used for training the model.
        Default is None.
    decimal : int, optional
        The number of decimal places to display in the output and plots.
        Default is 4.
    bins : int, optional
        The number of bins for the predicted probabilities histogram when
        `bin_strategy` is None. Default is 10.
    bin_strategy : str, optional
        The strategy for determining the number of bins for the predicted
        probabilities histogram. Can be 'sqrt', 'sturges', 'rice', 'freed',
        'scott', or 'doane'. Default is None.
    plot : bool, optional
        Whether to display the evaluation plots. Default is False.
    figsize : Tuple[int, int], optional
        The figure size for the plots in inches. Default is (12, 11).
    figmulti : float, optional
        The multiplier for the figure size in multi-class classification.
        Default is 1.7.
    conf_fontsize : int, optional
        The font size for the numbers in the confusion matrix. Default is 14.
    return_metrics : bool, optional
        Whether to return the evaluation metrics as a dictionary.
        Default is False.
    output : bool, optional
        Whether to print the evaluation results. Default is True.
    debug : bool, optional
        Whether to print debug information. Default is False.

    Returns
    -------
    metrics : Dict[str, Union[int, float]], optional
        A dictionary containing the evaluation metrics. Returned only if
        `return_metrics` is True and the classification type is binary.

    Examples
    --------
    Prepare data and model for the examples:

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.4, 0.6],
    ...                            random_state=42)
    >>> class_map = {0: 'Malignant', 1: 'Benign'}
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    ...                                                     random_state=42)
    >>> model = SVC(kernel='linear', probability=True, random_state=42)
    >>> model.fit(X_train, y_train)
    SVC(kernel='linear', probability=True, random_state=42)
    >>> y_pred = model.predict(X_test)

    Example 1: Basic evaluation with default settings:

    >>> eval_model(y_test=y_test, y_pred=y_pred)  #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       0.76      0.74      0.75        72
               1       0.85      0.87      0.86       128
    <BLANKLINE>
        accuracy                           0.82       200
       macro avg       0.81      0.80      0.80       200
    weighted avg       0.82      0.82      0.82       200
    <BLANKLINE>
                   Predicted:0         1
    Actual: 0                53        19
    Actual: 1                17        111
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.87
    True Negative Rate / Specificity: 0.74
    False Positive Rate / Fall-out: 0.26
    False Negative Rate / Miss Rate: 0.13
    <BLANKLINE>
    Positive Class: 1 (1)
    Threshold: 0.5

    Example 2: Evaluation with custom settings:

    >>> eval_model(y_test=y_test, y_pred=y_pred, estimator=model, x_test=X_test,
    ...            class_type='binary', class_map=class_map, pos_label=0,
    ...            threshold=0.35, model_name='SVM', class_weight='balanced',
    ...            decimal=4, plot=True, figsize=(13, 13), conf_fontsize=18,
    ...            bins=20)   #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    SVM Binary Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
          Benign     0.9545    0.8203    0.8824       128
       Malignant     0.7444    0.9306    0.8272        72
    <BLANKLINE>
        accuracy                         0.8600       200
       macro avg     0.8495    0.8754    0.8548       200
    weighted avg     0.8789    0.8600    0.8625       200
    <BLANKLINE>
    ROC AUC: 0.9220
    <BLANKLINE>
                   Predicted:1         0
    Actual: 1                105       23
    Actual: 0                5         67
    <BLANKLINE>
    True Positive Rate / Sensitivity: 0.9306
    True Negative Rate / Specificity: 0.8203
    False Positive Rate / Fall-out: 0.1797
    False Negative Rate / Miss Rate: 0.0694
    <BLANKLINE>
    Positive Class: Malignant (0)
    Class Weight: balanced
    Threshold: 0.35

    Example 3: Evaluate model with no output and return a dictionary:

    >>> metrics = eval_model(y_test=y_test, y_pred=y_pred, estimator=model,
    ...            x_test=X_test, class_map=class_map, pos_label=0,
    ...            return_metrics=True, output=False)
    >>> print(metrics)
    {'True Positives': 53, 'False Positives': 17, 'True Negatives': 111, 'False Negatives': 19, 'TPR': 0.7361111111111112, 'TNR': 0.8671875, 'FPR': 0.1328125, 'FNR': 0.2638888888888889, 'Benign': {'precision': 0.8538461538461538, 'recall': 0.8671875, 'f1-score': 0.8604651162790697, 'support': 128.0}, 'Malignant': {'precision': 0.7571428571428571, 'recall': 0.7361111111111112, 'f1-score': 0.7464788732394366, 'support': 72.0}, 'accuracy': 0.82, 'macro avg': {'precision': 0.8054945054945055, 'recall': 0.8016493055555556, 'f1-score': 0.8034719947592532, 'support': 200.0}, 'weighted avg': {'precision': 0.819032967032967, 'recall': 0.82, 'f1-score': 0.819430068784802, 'support': 200.0}, 'ROC AUC': 0.9219835069444444, 'Threshold': 0.5, 'Class Type': 'binary', 'Class Map': {0: 'Malignant', 1: 'Benign'}, 'Positive Label': 0, 'Title': None, 'Model Name': 'Model', 'Class Weight': None, 'Multi-Class': 'ovr', 'Average': 'macro'}

    Prepare multi-class example data:

    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> X = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length',
    ...                              'petal_width'])
    >>> y = pd.Series(y)
    >>> class_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    ...                                    random_state=42)
    >>> model = SVC(kernel='linear', probability=True, random_state=42)
    >>> model.fit(X_train, y_train)
    SVC(kernel='linear', probability=True, random_state=42)
    >>> y_pred = model.predict(X_test)

    Example 4: Evaluate multi-class model with default settings:

    >>> metrics = eval_model(y_test=y_test, y_pred=y_pred, class_map=class_map,
    ...               return_metrics=True)   #doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    Multi-Class Classification Report
    <BLANKLINE>
                  precision    recall  f1-score   support
    <BLANKLINE>
          Setosa       1.00      1.00      1.00        10
      Versicolor       1.00      1.00      1.00         9
       Virginica       1.00      1.00      1.00        11
    <BLANKLINE>
        accuracy                           1.00        30
       macro avg       1.00      1.00      1.00        30
    weighted avg       1.00      1.00      1.00        30
    <BLANKLINE>
    Predicted   Setosa  Versicolor  Virginica
    Actual
    Setosa          10           0          0
    Versicolor       0           9          0
    Virginica        0           0         11
    <BLANKLINE>
    >>> print(metrics)
    {'Setosa': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 10.0}, 'Versicolor': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 9.0}, 'Virginica': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 11.0}, 'accuracy': 1.0, 'macro avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 30.0}, 'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 30.0}, 'ROC AUC': None, 'Threshold': 0.5, 'Class Type': 'multi', 'Class Map': {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}, 'Positive Label': None, 'Title': None, 'Model Name': 'Model', 'Class Weight': None, 'Multi-Class': 'ovr', 'Average': 'macro'}
    """
    # Initialize debugging, controlled via 'debug' parameter
    db = DebugPrinter(debug = debug)
    db.print('-' * 40)
    db.print('START eval_model')
    db.print('-' * 40, '\n')
    db.print('y_test shape:', y_test.shape)
    db.print('y_pred shape:', y_pred.shape)
    db.print('class_map:', class_map)
    db.print('pos_label:', pos_label)
    db.print('class_type:', class_type)
    db.print('estimator:', estimator)
    if x_test is not None:
        db.print('x_test shape:', x_test.shape)
    else:
        db.print('x_test:', x_test)
    db.print('threshold:', threshold)

    # Convert y_test DataFrame to a Series if it's not already
    if isinstance(y_test, pd.DataFrame):
        db.print('\nConverting y_test DataFrame to Series...')
        db.print('y_test shape before:', y_test.shape)
        y_test = y_test.squeeze()
        db.print('y_test shape after:', y_test.shape)

    # Convert y_test DataFrame to a Series if it's not already
    if isinstance(y_pred, pd.DataFrame):
        db.print('\nConverting y_pred DataFrame to Series...')
        db.print('y_pred shape before:', y_pred.shape)
        y_pred = y_pred.squeeze()
        db.print('y_pred shape after:', y_pred.shape)

    # Get the unique labels and display labels for the confusion matrix
    if class_map is not None:
        # Make sure class_map is a dictionary
        if isinstance(class_map, dict):
            db.print('\nGetting labels from class_map...')
            unique_labels = list(class_map.keys())
            display_labels = list(class_map.values())
        else:
            raise TypeError("class_map must be a dictionary")

        # Make sure every unique_label has a corresponding entry in y_test
        missing_labels = set(np.unique(y_test)) - set(unique_labels)
        if missing_labels:
            db.print('y_test[:5]:', list(y_test[:5]))
            db.print('set(unique_labels):', set(unique_labels))
            db.print('set(np.unique(y_test)):', set(np.unique(y_test)))
            db.print('missing_labels:', missing_labels)
            raise ValueError(f"The following labels in y_test are missing from class_map: {missing_labels}")
    else:
        db.print('\nGetting labels from unique values in y_test...')
        unique_labels = np.unique(y_test)
        display_labels = [str(label) for label in unique_labels]
        db.print('Creating class_map...')
        class_map = {label: str(label) for label in unique_labels}
        db.print('class_map:', class_map)
    db.print('unique_labels:', unique_labels)
    db.print('display_labels:', display_labels)

    # Count the number of classes
    num_classes = len(unique_labels)
    db.print('num_classes:', num_classes)

    # If class_type is not passed, auto-detect based on unique values of y_test
    if class_type is None:
        if num_classes > 2:
            class_type = 'multi'
        elif num_classes == 2:
            class_type = 'binary'
        else:
            raise ValueError(f"Check data, cannot classify. Number of classes in y_test ({num_classes}) is less than 2: {unique_labels}")
        db.print(f"\nClassification type detected: {class_type}")
        db.print("Unique values in y:", num_classes)
    elif class_type not in ['binary', 'multi']:
        # If class type is invalid, raise an error
        raise ValueError(f"Class type '{class_type}' is invalid, must be 'binary' or 'multi'. Number of classes in y_test: {num_classes}, unique labels: {unique_labels}")

    # Check to ensure num_classes matches the passed class_type
    if class_type == 'binary' and num_classes != 2:
        raise ValueError(f"Class type is {class_type}, but number of classes in y_test ({num_classes}) is not 2: {unique_labels}")
    elif class_type == 'multi' and num_classes < 3:
        raise ValueError(f"Class type is {class_type}, but number of classes in y_test ({num_classes}) is less than 3: {unique_labels}")
    elif num_classes < 2:
        raise ValueError(f"Check data, cannot classify. Class type is {class_type}, and number of classes in y_test ({num_classes}) is less than 2: {unique_labels}")

    # Evaluation for multi-class classification
    if class_type == 'multi':

        # Set pos_label to None for multi-class
        pos_label = None

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Run the classification report
        db.print('\nRun the Classification Report...')
        class_report = classification_report(y_test, y_pred, digits=decimal, target_names=display_labels,
                                             zero_division=0, output_dict=True)
        db.print('class_report:', class_report)

        # Calculate ROC AUC if we have x_test and estimator
        if x_test is not None and estimator is not None:
            db.print('\nCalculating ROC AUC...')
            roc_auc = roc_auc_score(y_test, estimator.predict_proba(x_test), multi_class=multi_class, average=average)
        else:
            roc_auc = None
        db.print('roc_auc:', roc_auc)

        if output:
            # Display the best title we can create
            if title is not None:
                print(f"\n{title}\n")
            elif model_name != 'Model':
                print(f"\n{model_name} Multi-Class Classification Report\n")
            else:
                print(f"\nMulti-Class Classification Report\n")
            # Display the classification report
            print(classification_report(y_test, y_pred, digits=decimal, target_names=display_labels, zero_division=0))

            # Display the ROC AUC
            if roc_auc is not None:
                if isinstance(roc_auc, float):
                    print(f'ROC AUC: {round(roc_auc, decimal)}\n')
                elif isinstance(roc_auc, np.ndarray):
                    # It's an array, handle different cases
                    if roc_auc.size == 1:
                        print(f'ROC AUC: {round(roc_auc[0], decimal)}\n')
                    else:
                        # If it's an array with multiple elements, print the mean value, rounded
                        mean_roc_auc = np.mean(roc_auc)
                        print(f'ROC AUC (mean): {round(mean_roc_auc, decimal)}\n')
                else:
                    # Print it raw
                    print(f'ROC AUC: {roc_auc}\n')

            # Display the class weight for reference only
            if class_weight is not None:
                print(f'Class Weight: {class_weight}\n')

            # Create a DataFrame from the confusion matrix
            df_cm = pd.DataFrame(cm, index=display_labels, columns=display_labels)
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            print(f'{df_cm}\n')

    # Pre-processing for binary classification
    if class_type == 'binary':

        # Check if pos_label is in unique_labels
        if pos_label not in unique_labels:
            db.print('pos_label:', pos_label)
            db.print('type(pos_label):', type(pos_label).__name__)
            db.print('unique_labels:', unique_labels)
            db.print('unique_labels[0]:', unique_labels[0])
            db.print('unique_labels[1]:', unique_labels[1])
            db.print('type(unique_labels[0]):', type(unique_labels[0]).__name__)
            db.print('type(unique_labels[1]):', type(unique_labels[1]).__name__)
            raise ValueError(f"Positive label: {pos_label} ({type(pos_label).__name__}) is not in y_test unique values: {unique_labels}. Please specify the correct 'pos_label'.")

        # Encode labels if binary classification problem
        db.print('\nEncoding labels for binary classification...')

        # Assign neg_label based on pos_label
        neg_label = np.setdiff1d(unique_labels, [pos_label])[0]
        db.print('pos_label:', pos_label)
        db.print('neg_label:', neg_label)

        # Create a label_map for encoding
        label_map = {neg_label: 0, pos_label: 1}
        db.print('label_map:', label_map)

        # Encode new labels as 0 and 1
        db.print('\nEncoding y_test and y_pred...')
        y_test_enc = np.array([label_map[label] for label in y_test])
        y_pred_enc = np.array([label_map[label] for label in y_pred])
        db.print('y_test[:5]:', list(y_test[:5]))
        db.print('y_test_enc[:5]:', y_test_enc[:5])
        db.print('y_pred[:5]:', y_pred[:5])
        db.print('y_pred_enc[:5]:', y_pred_enc[:5])
        db.print('Overwriting y_test and y_pred...')
        y_test = y_test_enc
        y_pred = y_pred_enc
        db.print('y_test[:5]:', list(y_test[:5]))
        db.print('y_pred[:5]:', y_pred[:5])

        # Create a map for the new labels
        db.print('\nGetting the display labels...')
        pos_display = class_map[pos_label]
        neg_display = class_map[neg_label]
        db.print('pos_display:', pos_display)
        db.print('neg_display:', neg_display)
        if class_map is not None:
            display_map = {0: neg_display, 1: pos_display}
        else:
            display_map = {0: str(neg_label), 1: str(pos_label)}
        db.print('display_map:', display_map)

        # Update the unique labels and display labels for the confusion matrix
        db.print('\nUpdating labels from display_map...')
        unique_labels = list(display_map.keys())
        display_labels = list(display_map.values())
        db.print('New unique_labels:', unique_labels)
        db.print('New display_labels:', display_labels)

    # Calculate the probabilities
    if class_type == 'binary' and x_test is not None and estimator is not None:
        db.print('\nCalculating probabilities...')
        if hasattr(estimator, 'predict_proba'):
            probabilities = estimator.predict_proba(x_test)
            if probabilities.shape[1] == 2:
                probabilities = probabilities[:, 1]  # Use the probability of the positive class
            else:
                probabilities = probabilities.flatten()
        else:
            # If predict_proba is not available, use the raw predictions
            probabilities = estimator.predict(x_test).flatten()
        
        db.print('probabilities[:5]:', probabilities[:5])
        db.print('probabilities shape:', np.shape(probabilities))

        # Apply the threshold to the probabilities
        if plot or threshold != 0.5:
            db.print(f'\nApplying threshold {threshold} to probabilities...')
            y_pred_thresh = (probabilities >= threshold).astype(int)
            db.print('y_pred[:5]:', y_pred[:5])
            db.print('y_pred_thresh[:5]:', y_pred_thresh[:5])
            db.print('Overwriting y_pred with y_pred_thres...')
            y_pred = y_pred_thresh
            db.print('y_pred[:5]:', y_pred[:5])
        else:
            db.print(f'\nUsing default threshold of {threshold}...')
        db.print('plot:', plot)
    else:
        probabilities = None
        db.print(f'\nSkipping probabilities. class_type: {class_type}, x_test shape: {np.shape(x_test)}, estimator: {estimator.__class__.__name__}')

    # Evaluation for binary classification
    if class_type == 'binary':
        if output:
            # Display the best title we can create
            if title is not None:
                print(f"\n{title}\n")
            elif model_name != 'Model':
                print(f"\n{model_name} Binary Classification Report\n")
            else:
                print(f"\nBinary Classification Report\n")

        # Run the classification report
        db.print('\nRun the Classification Report...')
        class_report = classification_report(y_test, y_pred, labels=unique_labels, target_names=display_labels,
                                             digits=decimal, zero_division=0, output_dict=True)
        db.print('class_report:', class_report)
        if output:
            print(classification_report(y_test, y_pred, labels=unique_labels, target_names=display_labels,
                                        digits=decimal, zero_division=0))

        # Calculate the confusion matrix
        db.print('\nCalculating confusion matrix and metrics...')
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

        # Calculate the binary metrics
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tnr = tn / (tn + fp)
        fnr = fn / (fn + tp)
        db.print('cm:\n', cm)
        db.print('\ncm.ravel:', cm.ravel())
        db.print(f'TN: {tn}')
        db.print(f'FP: {fp}')
        db.print(f'FN: {fn}')
        db.print(f'TP: {tp}')

        binary_metrics = {
            "True Positives": tp,
            "False Positives": fp,
            "True Negatives": tn,
            "False Negatives": fn,
            "TPR": tpr,
            "TNR": tnr,
            "FPR": fpr,
            "FNR": fnr,
        }

    # Calculate the ROC AUC score if binary classification with probabilities
    if class_type == 'binary' and probabilities is not None:

        # Calculate ROC AUC score
        db.print('\nCalculating ROC AUC score...')
        roc_auc = roc_auc_score(y_test, probabilities, labels=unique_labels)
        db.print('y_test[:5]:', y_test[:5])
        db.print('probabilities[:5]:', probabilities[:5])
        db.print('unique_labels:', unique_labels)
        if output:
            print(f'ROC AUC: {roc_auc:.{decimal}f}\n')

        # Calculate false positive rate, true positive rate, and thresholds for ROC curve
        db.print('\nCalculating ROC curve...')
        fpr_array, tpr_array, thresholds = roc_curve(y_test, probabilities, pos_label=1)
        if len(thresholds) == 0 or len(fpr_array) == 0 or len(tpr_array) == 0:
            raise ValueError(f"Error in ROC curve calculation, at least one empty array. fpr_array length: {len(fpr_array)}, tpr_array length: {len(tpr_array)}, thresholds length: {len(thresholds)}.")
        db.print('y_test[:5]:', y_test[:5])
        db.print('probabilities[:5]:', probabilities[:5])
        db.print('Arrays from roc_curve:')
        db.print('fpr_array[:5]:', fpr_array[:5])
        db.print('tpr_array[:5]:', tpr_array[:5])
        db.print('thresholds[:5]:', thresholds[:5])

    # Print the binary classification output
    if class_type == 'binary' and output:

        # Print confusion matrix with display labels
        print(f"{'':<15}{'Predicted:':<10}{neg_display:<10}{pos_display:<10}")
        print(f"{'Actual: ' + neg_display:<25}{cm[0][0]:<10}{cm[0][1]:<10}")
        print(f"{'Actual: ' + pos_display:<25}{cm[1][0]:<10}{cm[1][1]:<10}")

        # Print evaluation metrics
        print("\nTrue Positive Rate / Sensitivity:", round(tpr, decimal))
        print("True Negative Rate / Specificity:", round(tnr, decimal))
        print("False Positive Rate / Fall-out:", round(fpr, decimal))
        print("False Negative Rate / Miss Rate:", round(fnr, decimal))
        print(f"\nPositive Class: {pos_display} ({pos_label})")
        if class_weight is not None:
            print("Class Weight:", class_weight)
        print("Threshold:", threshold)

    # Plot the evaluation metrics
    if plot or save_plots:

        # Create save directory if it doesn't exist
        if save_plots and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Define a blue color for plots
        blue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)

        # Just plot a confusion matrix for multi-class
        if class_type == 'multi':

            # Calculate the figure size for multi-class plots
            multiplier = figmulti
            max_size = 20
            size = min(len(unique_labels) * multiplier, max_size)
            figsize = (size, size)

            # Create a figure and axis for multi-class confusion matrix
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

            # Plot the confusion matrix
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            cm_display.plot(cmap='Blues', ax=ax1, colorbar=False)
            for text in cm_display.text_:
                for t in text:
                    t.set_fontsize(conf_fontsize - 2)  # Reduce font size for multi-class
            ax1.set_title(f'Confusion Matrix', fontsize=18, pad=15)
            ax1.set_xlabel('Predicted Label', fontsize=14, labelpad=15)
            ax1.set_ylabel('True Label', fontsize=14, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=10)

            plt.tight_layout()
            if save_plots:
                plt.savefig(os.path.join(save_dir, f'confusion_matrix_{timestamp}.png'))
                if wandb_run is not None:
                    wandb_run.log({f'confusion_matrix_{timestamp}': wandb.Image(plt)})
            if plot:
                plt.show()
            plt.close()

        # Just plot a confusion matrix for binary classification without probabilities
        elif class_type == 'binary' and probabilities is None:

            # Calculate the figure size for a single-chart plot
            multiplier = figmulti
            max_size = 20
            size = min(len(unique_labels) * multiplier, max_size) + 1.5  # Extra size for just 2 classes
            figsize = (size, size)

            # Create a figure and axis for a confusion matrix
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

            # Plot the confusion matrix
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            cm_display.plot(cmap='Blues', ax=ax1, colorbar=False)
            for text in cm_display.text_:
                for t in text:
                    t.set_fontsize(conf_fontsize)
            ax1.set_title(f'Confusion Matrix', fontsize=18, pad=15)
            ax1.set_xlabel('Predicted Label', fontsize=14, labelpad=15)
            ax1.set_ylabel('True Label', fontsize=14, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=10)

            plt.tight_layout()
            if save_plots:
                plt.savefig(os.path.join(save_dir, f'confusion_matrix_{timestamp}.png'))
                if wandb_run is not None:
                    wandb_run.log({f'confusion_matrix_{timestamp}': wandb.Image(plt)})
            if plot:
                plt.show()
            plt.close()

        # Plot 4 charts for binary classification
        elif class_type == 'binary' and probabilities is not None:

            # Calculate the number of bins
            if bin_strategy is not None:
                # Calculate the number of bins based on the specified strategy
                data_len = len(probabilities)
                if bin_strategy == 'sqrt':
                    num_bins = int(np.sqrt(data_len))
                elif bin_strategy == 'sturges':
                    num_bins = int(np.ceil(np.log2(data_len)) + 1)
                elif bin_strategy == 'rice':
                    num_bins = int(2 * data_len ** (1/3))
                elif bin_strategy == 'freed':
                    iqr = np.subtract(*np.percentile(probabilities, [75, 25]))
                    bin_width = 2 * iqr * data_len ** (-1/3)
                    num_bins = int(np.ceil((probabilities.max() - probabilities.min()) / bin_width))
                elif bin_strategy == 'scott':
                    std_dev = np.std(probabilities)
                    bin_width = 3.5 * std_dev * data_len ** (-1/3)
                    num_bins = int(np.ceil((probabilities.max() - probabilities.min()) / bin_width))
                elif bin_strategy == 'doane':
                    std_dev = np.std(probabilities)
                    skewness = ((np.mean(probabilities) - np.median(probabilities)) / std_dev)
                    sigma_g1 = np.sqrt(6 * (data_len - 2) / ((data_len + 1) * (data_len + 3)))
                    num_bins = int(np.ceil(np.log2(data_len) + 1 + np.log2(1 + abs(skewness) / sigma_g1)))
                else:
                    raise ValueError("Invalid bin strategy, possible values of 'bin_strategy' are 'sqrt', 'sturges', 'rice', 'freed', 'scott', and 'doane'")
            else:
                # Use default behavior of bins=10 for X axis range of 0 to 1.0
                num_bins = bins

            # Create a figure and subplots for binary classification plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

            # 1. Confusion Matrix
            cm_matrix = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, labels=unique_labels,
                                                                display_labels=display_labels, cmap='Blues', colorbar=False, normalize=None, ax=ax1)
            for text in cm_matrix.text_:
                for t in text:
                    t.set_fontsize(conf_fontsize)
            ax1.set_title(f'Confusion Matrix', fontsize=18, pad=15)
            ax1.set_xlabel('Predicted Label', fontsize=14, labelpad=15)
            ax1.set_ylabel('True Label', fontsize=14, labelpad=10)
            ax1.tick_params(axis='both', which='major', labelsize=11)

            # 2. Histogram of Predicted Probabilities
            ax2.hist(probabilities, color=blue, edgecolor='black', alpha=0.7, bins=num_bins, label=f'{model_name} Probabilities')
            ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold: {threshold:.{decimal}f}')
            ax2.set_title('Histogram of Predicted Probabilities', fontsize=18, pad=15)
            ax2.set_xlabel('Probability', fontsize=14, labelpad=15)
            ax2.set_ylabel('Frequency', fontsize=14, labelpad=10)
            ax2.set_xticks(np.arange(0, 1.1, 0.1))
            ax2.legend()

            # 3. ROC Curve
            ax3.plot([0, 1], [0, 1], color='grey', linestyle=':', label='Chance Baseline')
            ax3.plot(fpr_array, tpr_array, color=blue, marker='.', lw=2, label=f'{model_name} ROC Curve')
            ax3.scatter(fpr, tpr, color='red', s=80, zorder=5, label=f'Threshold {threshold:.{decimal}f}')
            ax3.axvline(x=fpr, ymax=tpr-0.027, color='red', linestyle='--', lw=1,
                        label=f'TPR: {tpr:.{decimal}f}, FPR: {fpr:.{decimal}f}')
            ax3.axhline(y=tpr, xmax=fpr+0.04, color='red', linestyle='--', lw=1)
            ax3.set_xticks(np.arange(0, 1.1, 0.1))
            ax3.set_yticks(np.arange(0, 1.1, 0.1))
            ax3.set_ylim(0,1.05)
            ax3.set_xlim(-0.05,1.0)
            ax3.grid(which='both', color='lightgrey', linewidth=0.5)
            ax3.set_title('ROC Curve', fontsize=18, pad=15)
            ax3.set_xlabel('False Positive Rate', fontsize=14, labelpad=15)
            ax3.set_ylabel('True Positive Rate', fontsize=14, labelpad=10)
            ax3.legend(loc='lower right')

            # 4. Precision-Recall Curve
            db.print('\nCalculating precision-recall curve...')
            db.print('y_test[:5]:', y_test[:5])
            db.print('probabilities[:5]:', probabilities[:5])
            db.print('pos_label:', pos_label)
            precision_array, recall_array, _ = precision_recall_curve(y_test, probabilities, pos_label=1)
            db.print('precision_array[:5]:', precision_array[:5])
            db.print('recall_array[:5]:', recall_array[:5])
            precision = class_report[pos_display]['precision']
            recall = class_report[pos_display]['recall']
            db.print('precision:', precision)
            db.print('recall:', recall)

            # Plot the Precision-Recall curve
            ax4.plot(recall_array, precision_array, marker='.', label=f'{model_name} Precision-Recall', color=blue)
            ax4.scatter(recall, precision, color='red', s=80, zorder=5, label=f'Threshold: {threshold:.{decimal}f}')
            ax4.axvline(x=recall, ymax=precision-0.025, color='red', linestyle='--', lw=1,
                        label=f'Precision: {precision:.{decimal}f}, Recall: {recall:.{decimal}f}')
            ax4.axhline(y=precision, xmax=recall-0.025, color='red', linestyle='--', lw=1)
            ax4.set_xticks(np.arange(0, 1.1, 0.1))
            ax4.set_yticks(np.arange(0, 1.1, 0.1))
            ax4.set_ylim(0,1.05)
            ax4.set_xlim(0,1.05)
            ax4.grid(which='both', color='lightgrey', linewidth=0.5)
            ax4.set_title('Precision-Recall Curve', fontsize=18, pad=15)
            ax4.set_xlabel('Recall', fontsize=14, labelpad=15)
            ax4.set_ylabel('Precision', fontsize=14, labelpad=10)
            ax4.legend(loc='lower left')

            plt.tight_layout()
            if save_plots:
                plt.savefig(os.path.join(save_dir, f'binary_class_plots_{timestamp}.png'))
                if wandb_run is not None:
                    wandb_run.log({f'binary_class_plots_{timestamp}.png': wandb.Image(plt)})
            if plot:
                plt.show()
            plt.close()

    # Package up the metrics if requested
    if return_metrics:

        # Custom metrics dictionary
        db.print('\nPackaging metrics dictionary...')
        custom_metrics = {
            "ROC AUC": roc_auc,
            "Threshold": threshold,
            #"Class Type": class_type,
            #"Class Map": class_map,
            #"Positive Label": pos_label,
            #"Title": title,
            #"Model Name": model_name,
            #"Class Weight": class_weight,
            #"Multi-Class": multi_class,
            #"Average": average
        }

        # Assemble the final metrics based on class type
        if class_type == 'binary':
            metrics = {**binary_metrics, **class_report, **custom_metrics}
        else:
            metrics = {**class_report, **custom_metrics}
        db.print('metrics:', metrics)

        # Return a dictionary of metrics
        return metrics

