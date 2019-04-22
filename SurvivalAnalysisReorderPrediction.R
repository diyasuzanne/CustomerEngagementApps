# An interesting business problem is not only predicting if a customer will purchase another product again 
# but also understanding how soon the customer will purchase another product

# Survival Analysis: Model when an event will take place not just if an event will take place


# Import packages

library(ggplot2)
library(Hmisc)
library(survival)
library(rms)

# Reading in the dataset
  CustomerReorder <- read.csv("CustomerReorder.csv")
  head(CustomerReorder)

# Data Understanding
  # DaysSinceFirstPurch: The difference between the current day and first purchase, if repeat purchase then the difference between the first and second purchase (uncensored event)

# Exploratory analysis    
    # Compare the distribution of days since first purchase for repeat versus customers with only single purchases
        ggplot(CustomerReorder) + geom_histogram(aes(x = CustomerReorder$daysSinceFirstPurch, fill = factor(CustomerReorder$boughtAgain))) + facet_grid( ~ boughtAgain) 
            # Repeat customers cohort typically have more customers with lower days to purchase


    
    
# Creating the survival variable (target variable)
    CustomerReorder$SurVar <- Surv(CustomerReorder$daysSinceFirstPurch, CustomerReorder$boughtAgain)
    head(CustomerReorder)

    # Survival function probability that a customer will not churn during the time period specified
    
    # Hazard function: probability of churn happening given that the event has not happened yet
    
    # Kaplan Meier with the survival object as the target variable
    
    
    SurKM <- survfit(CustomerReorder$SurVar ~ 1, type = "kaplan-meier")  # Not considering any independent variable
    SurKM$surv
    
    print(SurKM)
      # Results
        # 63% (3199 out of 5122) customers churned or reordered
        # Median number of days it took to reorder are 41
    
    plot(SurKM) # Survival function and the confidence interval
    
    
## Adding covariates to understand the impact of those variables
    
    SurKMGender <- survfit(SurVar ~ gender, data = CustomerReorder)
    SurKMVoucher <- survfit(SurVar ~ voucher, data = CustomerReorder)
    SurKMReturned <- survfit(SurVar ~ returned, data = CustomerReorder)
    
    print(SurKMGender) # Women tend to reorder more but take longer to reorder
    print(SurKMVoucher)  # Customers without vouchers are faster to reorder
    print(SurKMReturned) # Customers without returns tend to reorder more and do so faster
    
    
    units(CustomerReorder$daysSinceFirstPurch) <- "Month"
    head(CustomerReorder)
    
    # Multi-variate survival analysis
    dd <- datadist(CustomerReorder) 
    options(datadist = "dd")  
    
        SurvCox <- cph(SurVar ~ gender + voucher + returned + shoppingCartValue,
                       data = CustomerReorder, x = TRUE, y = TRUE, surv = TRUE, time.inc = 1)
    
    print(SurvCox)        
    SurvCox$coefficients
    exp(SurvCox$coefficients)
      
    
    plot(summary(SurvCox), log = TRUE)
    
    
    
    ## Checking assumptions for proportioanl hazards -     
    testSurvCox <-  cox.zph(SurvCox)
    print(testSurvCox)
    
    plot(testSurvCox, var = "gender=male")
    
    SurvCox1 <- cph(SurVar ~  voucher + returned + shoppingCartValue,
                   data = CustomerReorder, x = TRUE, y = TRUE, surv = TRUE, time.inc = 1)
    
    testSurvCox1 <-  cox.zph(SurvCox1)
    print(testSurvCox1)
    
    
    # Validate model
    validate(SurvCox, method = "crossvalidation",B = 10, dxy = TRUE, pr = FALSE)
    
    validate(SurvCox1, method = "crossvalidation",B = 10, dxy = TRUE, pr = FALSE)  # Marginal increase in R2 but model can be improved with additional variables
    
    
    # Prediction
    CustomerN <- data.frame(daysSinceFirstPurch = 21, shoppingCartValue = 99.90, voucher = 1, returned = 0, stringsAsFactors = FALSE)
    
    # Make predictions
    pred <- survfit(SurvCox1, newdata = CustomerN)
    print(pred)

    # Predicted median time of 46 days for the customer to reorder
    
    