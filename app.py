import pickle
import numpy as np
import os
from flask import Flask,render_template,request
app=Flask( __name__ )
model2=pickle.load(open("model2.pkl", "rb"))

@app.route("/",methods=["GET","POST"])
def myFunc():
    prediction=None
    result_class=None
    if request.method=="POST":
        Age=float(request.form["age"])
        Gender=float(request.form['gender'])
        State=float(request.form["state"])
        ProductCategory=float(request.form["product"])
        TotalSpent=float(request.form["total"])
        NumOrders=float(request.form["NumOfOrders"])
        DaysSinceLastOrder=float(request.form["DaysLastActive"])
        NumComplaints=float(request.form["complaints"])
        DiscountsUsed=float(request.form["discountused"])
        SatisfactionScore=float(request.form["satisfaction"])
        DeliveryDays=float(request.form["delivery"])
        NumReturns=float(request.form["returns"])
        LoyaltyYears=float(request.form["loyalty"])
        LoyaltyIndex=LoyaltyYears*SatisfactionScore
        SatisfactionIndex=SatisfactionScore/(DeliveryDays+NumComplaints)
        features=np.array([[Age,Gender,State,ProductCategory,TotalSpent,NumOrders,DaysSinceLastOrder,NumComplaints,DiscountsUsed,SatisfactionScore,DeliveryDays,NumReturns,LoyaltyYears,SatisfactionIndex,LoyaltyIndex]])
        result=model2.predict(features)[0]
        if result==1:
            prediction="Churning"
            result_class="churning"
        else:
            prediction="Not Churning"
            result_class="notchurning"
    return render_template("front.html",answer=prediction,result_class=result_class)    
if __name__==("__main__"):
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)          

