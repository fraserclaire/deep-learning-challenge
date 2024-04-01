# Module 21 Challenge - Neural Networks and Deep Learning

## Analysis Overview:

Our goal is to create a deep learning model by identifying key features of the dataset and using these features to predict whether applications will be successful or not.

### Data Preprocessing:

- Target: 
    -** IS_SUCCESSFUL**: Was the money used effectively

- Features: 
    - **EIN, NAME**: Identification columns
    - **APPLICATION_TYPE**
    - **CLASSIFICATION**: Government organization classification
    - **USE_CASE**: Use case for funding
    - **ORGANIZATION**
    - **STATUS**: Active Status
    - **INCOME_AMT**
    - **SPECIAL_CONSIDERATIONS**
    - **ASK_AMT**


## Model Performance:

1. Exclude EIN, NAME; Reduced categories for APPLICATION_TYPE and CLASSIFICATION 

```
Checkpoint file: "AlphabetSoupCharity.h5"

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 80)                3520      
                                                                 
 dense_1 (Dense)             (None, 30)                2430      
                                                                 
 dense_2 (Dense)             (None, 1)                 31        
                                                                 
=================================================================
Total params: 5981 (23.36 KB)
Trainable params: 5981 (23.36 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
- Performance over 100 Epochs:
    - loss: 0.5580 
    - accuracy: 0.7235
```

2. Reduce number of units for Model 1 

Checkpoint file: "AlphabetSoupCharity_Optimization.h5"

Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_3 (Dense)             (None, 40)                1760      
                                                                 
 dense_4 (Dense)             (None, 10)                410       
                                                                 
 dense_5 (Dense)             (None, 1)                 11        
                                                                 
=================================================================
Total params: 2181 (8.52 KB)
Trainable params: 2181 (8.52 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
- Performance over 100 Epochs:
    - loss: 0.5545 
    - accuracy: 0.7236

The reduction of units from 80 -> 40 and 30 -> 10 for the first and second layers, respectively results in a slight improvement in model performance. The increase in accuracy and reduction of loss is very minimal here.

3. Model 1 with third hidden layer

Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_6 (Dense)             (None, 40)                1760      
                                                                 
 dense_7 (Dense)             (None, 10)                410       
                                                                 
 dense_8 (Dense)             (None, 2)                 22        
                                                                 
 dense_9 (Dense)             (None, 1)                 3         
                                                                 
=================================================================
Total params: 2195 (8.57 KB)
Trainable params: 2195 (8.57 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
- Performance over 100 Epochs:
    - loss: 0.5518 
    - accuracy: 0.7271

The addition of a third layer also slightly improves model performance.


4. Exclude EIN; Reduced categories for APPLICATION_TYPE, CLASSIFICATION, and NAME

Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_10 (Dense)            (None, 40)                17880     
                                                                 
 dense_11 (Dense)            (None, 10)                410       
                                                                 
 dense_12 (Dense)            (None, 2)                 22        
                                                                 
 dense_13 (Dense)            (None, 1)                 3         
                                                                 
=================================================================
Total params: 18315 (71.54 KB)
Trainable params: 18315 (71.54 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
- Performance over 100 Epochs:
    - loss: 0.4593 
    - accuracy: 0.7883

Including NAME (with reduced number of categories to group names with <5 requests) improves model performance by increasing accuracy by ~6% and reducing loss by ~10%.

5. Model 4 excluding AFFILIATION column

Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_14 (Dense)            (None, 40)                17640     
                                                                 
 dense_15 (Dense)            (None, 10)                410       
                                                                 
 dense_16 (Dense)            (None, 2)                 22        
                                                                 
 dense_17 (Dense)            (None, 1)                 3         
                                                                 
=================================================================
Total params: 18075 (70.61 KB)
Trainable params: 18075 (70.61 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
- Performance over 100 Epochs:
    - loss: 0.4532 
    - accuracy: 0.7887

The model performs slightly better after excluding the AFFILIATION column, which likely implies that these data do not inform whether a loan was successful after funding or not.

Similar model generation and testing on all columns may provide insight into which data are likely to inform accurate model predictions.
