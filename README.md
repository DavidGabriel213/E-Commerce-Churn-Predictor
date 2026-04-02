# Nigerian E-commerce Customer Churn Predictor

Predicts whether a Nigerian e-commerce customer will churn
based on transaction history, satisfaction and loyalty data.

Live App: https://e-commerce-churn-predictor-8b9b.onrender.com

## Dataset
- 1530 rows, 18 columns (extreme mess!)
- Target: Churned (Yes/No)

## Cleaning Highlights
- DeliveryDays: '4-5 days' → averaged to 4.5
- Churned: 12 different formats standardized
- TotalSpent: NGN/₦ currency formats stripped
- SatisfactionScore: '4/5' and '3 stars' parsed

## Feature Engineering
| Feature | Formula |
|---|---|
| SatisfactionIndex | Score / (Delivery + Complaints) |
| LoyaltyIndex | LoyaltyYears x SatisfactionScore |

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 90.7% |
| RF Tuned | 86.4% |
| Random Forest | 85.7% |
| Decision Tree | 82.9% |

## Stack
Python · Pandas · Scikit-learn · Flask · HTML · CSS · Render

---
Self-taught ML project — built on Android phone.
GitHub: github.com/DavidGabriel213
