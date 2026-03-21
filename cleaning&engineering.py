import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/storage/emulated/0/Download/nigerian_ecommerce_messy.csv")
df.drop_duplicates(inplace=True)
df["Age"]=df["Age"].astype(str).str.replace("years","").str.replace("-","").str.strip()
df["Age"]=pd.to_numeric(df['Age'],errors="coerce")
df["Age"]=df["Age"].fillna(df["Age"].median())
max1=df["Age"].quantile(0.75)+1.5*(df["Age"].quantile(0.75)-df["Age"].quantile(0.25))
min1=df["Age"].quantile(0.25)-1.5*(df["Age"].quantile(0.75)-df["Age"].quantile(0.25))
df["Age"]=df["Age"].clip(min1,max1)
df["Age"]=df["Age"].astype(int)
df["Gender"]=df["Gender"].astype(str).str.capitalize().str.strip()
gender_corrector={"F":"Female","M":"Male","Male":"Male","Female":"Female"}
df["Gender"]=df["Gender"].map(gender_corrector)
df["State"]=df["State"].astype(str).str.strip().str.capitalize()
df["DeviceType"]=df["DeviceType"].astype(str).str.capitalize().str.strip()
df["PaymentMethod"]=df["PaymentMethod"].astype(str).str.capitalize().str.strip()
payment_corrector={"Ussd":"USSD","Nan":np.nan,"Transfer":"Bank transfer","Card":"Debit card","Pos":"POS"}
df["PaymentMethod"]=df["PaymentMethod"].replace(payment_corrector)
df["PaymentMethod"]=df["PaymentMethod"].fillna(df.groupby(["State","DeviceType"])["PaymentMethod"].transform(lambda x : x.mode()[0]))
df["ProductCategory"]=df["ProductCategory"].astype(str).str.capitalize().str.strip()
product_corrector={"Nan":np.nan}
df["ProductCategory"]=df["ProductCategory"].replace(product_corrector)
df["ProductCategory"]=df["ProductCategory"].fillna(df.groupby(["Gender","State"])["ProductCategory"].transform(lambda x: x.mode()[0]))
df["TotalSpent(NGN)"]=df["TotalSpent(NGN)"].astype(str).str.replace("-","").str.replace("\u20a6", "").str.replace("NGN","").str.strip()
df["TotalSpent(NGN)"]=pd.to_numeric(df["TotalSpent(NGN)"],errors="coerce")
max2=df["TotalSpent(NGN)"].quantile(0.75)+1.5*(df["TotalSpent(NGN)"].quantile(0.75)-df["TotalSpent(NGN)"].quantile(0.25))
min2=df["TotalSpent(NGN)"].quantile(0.25)-1.5*(df["TotalSpent(NGN)"].quantile(0.75)-df["TotalSpent(NGN)"].quantile(0.25))
df["TotalSpent(NGN)"]=df["TotalSpent(NGN)"].clip(min2,max2)
df["TotalSpent(NGN)"]=(df["TotalSpent(NGN)"].fillna(df.groupby("ProductCategory")["TotalSpent(NGN)"].transform("mean"))).round(2)
max3=df["NumOrders"].quantile(0.75)+1.5*(df["NumOrders"].quantile(0.75)-df["NumOrders"].quantile(0.25))
min3=df["NumOrders"].quantile(0.25)-1.5*(df["NumOrders"].quantile(0.75)-df["NumOrders"].quantile(0.25))
df["NumOrders"]=df["NumOrders"].clip(min3,max3)
df["NumOrders"]=(df["NumOrders"].fillna(df["NumOrders"].median())).astype(int)
df["AvgOrderValue(NGN)"]=(df["TotalSpent(NGN)"]/df["NumOrders"]).round(2)
df["DaysSinceLastOrder"]=np.abs(df["DaysSinceLastOrder"])
df["NumComplaints"]=np.abs(df["NumComplaints"])
df["DiscountsUsed"]=np.abs(df["DiscountsUsed"])
df["SatisfactionScore"]=df["SatisfactionScore"].astype(str).str.replace("stars","").str.strip()
def satisfaction_correction(c):
    if "/" in c:
        k=c.index("/")
        return c[:k]
    elif c=="nan":
        return np.nan
    else:
        return c
df["SatisfactionScore"]=df["SatisfactionScore"].apply(lambda x: satisfaction_correction(x))
df["SatisfactionScore"]=np.abs(pd.to_numeric(df["SatisfactionScore"],errors="coerce"))
df["SatisfactionScore"]=df["SatisfactionScore"].clip(0,5)
df["SatisfactionScore"]=(df["SatisfactionScore"].fillna(df.groupby("NumComplaints")["SatisfactionScore"].transform("mean"))).round(1)
df["DeliveryDays"]=df["DeliveryDays"].astype(str).str.replace("days","").str.strip()
def delivery_correction(c):
    if isinstance(c, str) and '-' in c and not c.startswith('-'):
        parts = c.split('-')
        if all(p.isdigit() for p in parts):
            return str(np.ceil(sum(map(int, parts))/2)) 
    return c 
df["DeliveryDays"]=df["DeliveryDays"].apply(lambda x: delivery_correction(x))
df["DeliveryDays"]=np.abs(pd.to_numeric(df["DeliveryDays"],errors="coerce"))
df["DeliveryDays"]=(df["DeliveryDays"].fillna(df["DeliveryDays"].mode()[0])).astype(int)
df["NumReturns"]=np.abs(df["NumReturns"])
df["LoyaltyYears"]=df["LoyaltyYears"].astype(str).str.replace("yrs","").str.replace("years","").str.replace("-","").str.strip()
df["LoyaltyYears"]=pd.to_numeric(df["LoyaltyYears"],errors="coerce")
df["LoyaltyYears"]=df["LoyaltyYears"].fillna(df["LoyaltyYears"].mean()).round(1)
df["Churned"]=df["Churned"].astype(str).str.capitalize().str.strip()
churn_correction={"0":"No","1":"Yes","False":"No","True":"Yes","N":"No","Y":"Yes"}
df["Churned"]=df["Churned"].replace(churn_correction)
df["SatisfactionIndex"]=(df["SatisfactionScore"]/(df["DeliveryDays"]+df["NumComplaints"])).round(3)
df["LoyaltyIndex"]=(df["LoyaltyYears"]*df["SatisfactionScore"]).round(3)
df.to_csv("EcommerceClean.csv")