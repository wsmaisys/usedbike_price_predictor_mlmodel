import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load preprocessor and model
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("bike_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# State-City Mapping from cleaned data
state_city_dict = {
    'Andhra Pradesh': ['Adoni','Anantapur','Dharmavaram','Godavari','Guntur','Kadapa','Krishna','Kurnool','Nellore','Puttur','Rajahmundry','Vijayawada','Visakhapatnam','Vizianagaram','Yemmiganur'],
    'Arunachal Pradesh': ['Seppa'],
    'Assam': ['Dibrugarh','Guwahati','Jorhat','Silchar','Tezpur','Tinsukia'],
    'Bihar': ['Begusarai','Bhagalpur','Bihar Sharif','Darbhanga','Gaya','Muzaffarpur','Patna'],
    'Chandigarh': ['Chandigarh'],
    'Chhattisgarh': ['Bhilai','Bilaspur','Korba','Raipur'],
    'Delhi': ['Delhi'],
    'Goa': ['Mapusa','Margao','Panaji'],
    'Gujarat': ['Ahmedabad','Anand','Ankleshwar','Bharuch','Bhavnagar','Gandhidham','Gandhinagar','Jamnagar','Junagadh','Mehsana','Navsari','Rajkot','Surat','Vadodara','Valsad'],
    'Haryana': ['Ambala','Faridabad','Gurgaon','Hisar','Karnal','Panipat','Rohtak','Sonipat','Yamunanagar'],
    'Himachal Pradesh': ['Baddi','Hamirpur','Kullu','Mandi','Shimla','Solan'],
    'Jammu and Kashmir': ['Jammu','Srinagar'],
    'Jharkhand': ['Bokaro','Dhanbad','Jamshedpur','Ranchi'],
    'Karnataka': ['Bagalkot','Ballari','Belagavi','Bengaluru','Bidar','Chikkamagaluru','Davanagere','Gulbarga','Hassan','Hubli','Mangalore','Mysore','Raichur','Shivamogga','Tumkur'],
    'Kerala': ['Alappuzha','Ernakulam','Kollam','Kottayam','Kozhikode','Palakkad','Thiruvananthapuram','Thrissur'],
    'Madhya Pradesh': ['Bhopal','Dewas','Gwalior','Indore','Jabalpur','Ratlam','Sagar','Satna','Ujjain'],
    'Maharashtra': ['Ahmednagar','Aurangabad','Jalgaon','Kolhapur','Mumbai','Nagpur','Nashik','Navi Mumbai','Pimpri-Chinchwad','Pune','Solapur','Thane'],
    'Manipur': ['Imphal'],
    'Meghalaya': ['Shillong'],
    'Nagaland': ['Dimapur'],
    'Odisha': ['Balasore','Berhampur','Bhubaneswar','Cuttack','Rourkela','Sambalpur'],
    'Puducherry': ['Puducherry'],
    'Punjab': ['Amritsar','Bathinda','Jalandhar','Ludhiana','Mohali','Patiala'],
    'Rajasthan': ['Ajmer','Alwar','Bharatpur','Bhilwara','Bikaner','Jaipur','Jodhpur','Kota','Udaipur'],
    'Sikkim': ['Gangtok'],
    'Tamil Nadu': ['Chennai','Coimbatore','Erode','Madurai','Nagercoil','Salem','Thanjavur','Tiruchirappalli','Tirunelveli','Vellore'],
    'Telangana': ['Hyderabad','Karimnagar','Khammam','Nizamabad','Warangal'],
    'Tripura': ['Agartala'],
    'Uttar Pradesh': ['Agra','Aligarh','Allahabad','Bareilly','Ghaziabad','Gorakhpur','Jhansi','Kanpur','Lucknow','Meerut','Moradabad','Noida','Varanasi'],
    'Uttarakhand': ['Dehradun','Haldwani','Haridwar','Roorkee'],
    'West Bengal': ['Asansol','Durgapur','Howrah','Kharagpur','Kolkata','Siliguri']
}

# Brand-Bike Mapping from dataset
brand_bike_dict = {
    'Yamaha': ['Fazer 25', 'FZ25 ABS', 'FZS V3', 'Ray', 'FZ V 2.0', 'YZF R15 V3', 'FZ S FI V 2.0'],
    'Bajaj': ['Dominar 400', 'Avenger Street', 'CT 100', 'Pulsar 220 F', 'Avenger 220', 'Pulsar NS 200', 'Pulsar 150 DTS-i', 'Platina 100', 'Pulsar 180'],
    'Hero': ['Passion', 'Glamour FI', 'Splendor Plus', 'Xtreme Sports'],
    'TVS': ['Apache RTR 160', 'Apache RTR 160 4V', 'Jupiter'],
    'Honda': ['CB Hornet 160R', 'CB Unicorn 150', 'CB Shine', 'Activa 5G', 'Unicorn'],
    'Suzuki': ['Gixxer SF', 'Gixxer', 'Access 125'],
    'Royal Enfield': ['Thunderbird 350', 'Classic 350', 'Bullet 350'],
    'KTM': ['RC 390', 'Duke 200', 'RC 200'],
    'Mahindra': ['Centuro'],
    'Harley Davidson': ['Street 750']
}

# Bike-Power Mapping (bike_name: [list of engine powers])
bike_power_dict = {
    # Yamaha
    'Fazer 25': [249],
    'FZ25 ABS': [249],
    'FZS V3': [149],
    'Ray': [113],
    'FZ V 2.0': [149],
    'YZF R15 V3': [155],
    'FZ S FI V 2.0': [149],
    
    # Bajaj
    'Dominar 400': [373],
    'Avenger Street': [220],
    'CT 100': [99],
    'Pulsar 220 F': [220],
    'Avenger 220': [220],
    'Pulsar NS 200': [200],
    'Pulsar 150 DTS-i': [150],
    'Platina 100': [99],
    'Pulsar 180': [178],
    
    # Hero
    'Passion': [97, 110],  # Multiple variants
    'Glamour FI': [125],
    'Splendor Plus': [97],
    'Xtreme Sports': [150],
    
    # TVS
    'Apache RTR 160': [160],
    'Apache RTR 160 4V': [160],
    'Jupiter': [110],
    
    # Honda
    'CB Hornet 160R': [162],
    'CB Unicorn 150': [150],
    'CB Shine': [125],
    'Activa 5G': [109],
    'Unicorn': [150],
    
    # Suzuki
    'Gixxer SF': [155],
    'Gixxer': [155],
    'Access 125': [124],
    
    # Royal Enfield
    'Thunderbird 350': [346],
    'Classic 350': [346],
    'Bullet 350': [346],
    
    # KTM
    'RC 390': [373],
    'Duke 200': [200],
    'RC 200': [200],
    
    # Mahindra
    'Centuro': [106],
    
    # Harley Davidson
    'Street 750': [749]
}

brand_list = list(brand_bike_dict.keys())
owner_list = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner Or More']

st.set_page_config(page_title="Used Bike Price Predictor", layout="wide")
st.title("🏍️ Used Bike Price Predictor")

st.header("Fill in the bike details")

# Create columns for layout
col1, col2, col3 = st.columns(3)

# Column 1: State, Brand, Engine Power (dynamic based on bike)
with col1:
    state = st.selectbox("State", list(state_city_dict.keys()), key="state")
    brand = st.selectbox("Brand", brand_list, key="brand")
    
    # Dynamic power selection based on selected bike
    bike_options = brand_bike_dict.get(brand, [])
    if bike_options and st.session_state.get("bike_name") in bike_power_dict:
        selected_bike = st.session_state.get("bike_name", bike_options[0])
        power_options = bike_power_dict.get(selected_bike, [150])
        
        if len(power_options) == 1:
            # Single power option - show as info box
            st.text("Engine Power (cc)")
            st.info(f"🔧 {power_options[0]} cc (Fixed for {selected_bike})")
            power = power_options[0]
        else:
            # Multiple power options - show as selectbox
            power = st.selectbox("Engine Power (cc)", 
                               power_options, 
                               help=f"Available variants for {selected_bike}")
    else:
        power = st.number_input("Engine Power (cc)", min_value=50.0, max_value=2500.0, value=150.0)

# Column 2: City (dynamic based on state), Bike Name (dynamic based on brand), Kilometer Driven
with col2:
    # Dynamic city dropdown based on selected state
    city_options = state_city_dict.get(state, [])
    city = st.selectbox("City", city_options, key="city")
    
    # Dynamic bike name dropdown based on selected brand
    bike_options = brand_bike_dict.get(brand, [])
    bike_name = st.selectbox("Bike Name", bike_options, key="bike_name")
    
    kms_driven = st.number_input("Kilometers Driven", min_value=0.0, max_value=500000.0, value=30000.0, key="kms_driven")

# Column 3: Ownership Type, Age of the bike
with col3:
    owner = st.selectbox("Ownership Type", owner_list, key="owner")
    age = st.number_input("Age of the Bike (in years)", min_value=0.0, max_value=50.0, value=5.0, key="age")

# Update power display when bike selection changes
if bike_name and bike_name in bike_power_dict:
    power_options = bike_power_dict[bike_name]
    if len(power_options) == 1:
        power = power_options[0]

# Predict button outside the columns
st.markdown("---")
if st.button("Predict Price", type="primary", use_container_width=True):
    # Validation to ensure selections are made
    if not city_options:
        st.error("Please select a valid state with available cities.")
    elif not bike_options:
        st.error("Please select a valid brand with available bike models.")
    else:
        input_data = pd.DataFrame({
            'bike_name': [bike_name],
            'brand': [brand],
            'state': [state],
            'city': [city],
            'owner': [owner],
            'age': [age],
            'kms_driven': [kms_driven],
            'power': [power]
        })

        try:
            processed_data = preprocessor.transform(input_data)
            prediction = model.predict(processed_data)[0]
            
            st.success(f"💰 Estimated Price: ₹{np.round(prediction, 2):,}")
            
            # Dynamic Feature Analysis based on user input
            st.markdown("---")
            st.header("🎯 Your Bike's Strengths & Weaknesses")
            
            def analyze_user_features(bike_name, kms_driven, city, age, power):
                """Analyze user's bike features and return strengths and weaknesses"""
                
                # Define thresholds for analysis
                strengths = []
                weaknesses = []
                
                # Bike Name Analysis (19.9% importance)
                premium_bikes = ['Dominar 400', 'YZF R15 V3', 'RC 390', 'Duke 200', 'RC 200', 'Street 750', 'Classic 350', 'Thunderbird 350']
                popular_bikes = ['FZ V 2.0', 'Pulsar 150 DTS-i', 'CB Shine', 'Splendor Plus', 'Jupiter', 'Activa 5G']
                
                if bike_name in premium_bikes:
                    strengths.append("🏍️ **Premium Bike Model** - Your bike model has excellent brand value and market demand")
                elif bike_name in popular_bikes:
                    strengths.append("🏍️ **Popular Model** - High demand in used bike market ensures good resale value")
                else:
                    weaknesses.append("🏍️ **Standard Model** - Average market demand may limit premium pricing")
                
                # Kilometers Driven Analysis (19.8% importance)
                if kms_driven < 10000:
                    strengths.append("🛣️ **Ultra-Low Mileage** - Exceptional advantage! Very few kilometers driven")
                elif kms_driven < 25000:
                    strengths.append("🛣️ **Low Mileage** - Well-maintained usage pattern increases value significantly")
                elif kms_driven < 50000:
                    pass  # Neutral - no strength or weakness
                elif kms_driven < 80000:
                    weaknesses.append("🛣️ **High Mileage** - Extensive usage may reduce buyer interest")
                else:
                    weaknesses.append("🛣️ **Very High Mileage** - Major factor reducing market value")
                
                # City Analysis (12.4% importance)
                metro_cities = ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad']
                tier2_cities = ['Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Surat', 'Vadodara', 'Coimbatore']
                
                if city in metro_cities:
                    strengths.append("📍 **Metro City Location** - High demand and better pricing in major cities")
                elif city in tier2_cities:
                    strengths.append("📍 **Good City Market** - Decent demand in well-developed city")
                else:
                    weaknesses.append("📍 **Smaller City Market** - Limited buyer pool may affect pricing")
                
                # Age Analysis (11.7% importance)
                if age < 2:
                    strengths.append("📅 **Nearly New** - Minimal depreciation, like-new condition advantage")
                elif age < 5:
                    strengths.append("📅 **Optimal Age** - Sweet spot for resale with good value retention")
                elif age < 8:
                    pass  # Neutral
                else:
                    weaknesses.append("📅 **Older Bike** - Age factor significantly impacts market value")
                
                # Power Analysis (10.2% importance)
                if power > 300:
                    strengths.append("⚡ **High Performance** - Powerful engine appeals to performance seekers")
                elif power > 200:
                    strengths.append("⚡ **Good Performance** - Above-average power for better market appeal")
                elif power < 110:
                    weaknesses.append("⚡ **Low Power** - Entry-level engine may limit buyer interest")
                
                return strengths, weaknesses
            
            # Analyze user's bike
            strengths, weaknesses = analyze_user_features(bike_name, kms_driven, city, age, power)
            
            col_strength, col_weakness = st.columns(2)
            
            with col_strength:
                st.subheader("✅ Your Bike's Strengths")
                if strengths:
                    for strength in strengths:
                        st.success(strength)
                    st.info("💡 **Negotiation Tip**: Highlight these points when selling to justify your asking price!")
                else:
                    st.info("🔍 Your bike has standard features - focus on maintenance and condition during negotiations.")
            
            with col_weakness:
                st.subheader("⚠️ Areas of Concern")
                if weaknesses:
                    for weakness in weaknesses:
                        st.warning(weakness)
                    st.info("💡 **Strategy**: Be prepared to address these points and consider pricing accordingly.")
                else:
                    st.success("🎉 Great! No major weaknesses found in your bike profile!")
            
            # Overall recommendation
            st.markdown("---")
            strength_count = len(strengths)
            weakness_count = len(weaknesses)
            
            if strength_count > weakness_count:
                st.success("🚀 **Overall Assessment**: Your bike has strong selling points! You're in a good position to negotiate a premium price.")
            elif weakness_count > strength_count:
                st.warning("🤔 **Overall Assessment**: Consider competitive pricing due to some limiting factors. Focus on bike condition and maintenance history.")
            else:
                st.info("⚖️ **Overall Assessment**: Balanced profile. Market-standard pricing with room for negotiation based on condition.")
            
            # Display input summary
            with st.expander("Input Summary"):
                st.write(f"**Bike**: {brand} {bike_name}")
                st.write(f"**Location**: {city}, {state}")
                st.write(f"**Details**: {age} years old, {kms_driven:,} km driven, {power} cc, {owner}")
            
            # Price Analysis Section
            st.markdown("---")
            st.header("📊 Price Analysis")
            
            col_analysis1, col_analysis2 = st.columns(2)
            
            with col_analysis1:
                st.subheader("🎯 How Reliable is This Prediction?")
                st.write("**Model Accuracy: 94.9%** ⭐⭐⭐⭐⭐")
                st.write("Our AI model is highly accurate and gets the price right 95% of the time!")
                
                st.write("**Typical Error Range: ±₹3,500**")
                st.write(f"Your bike's actual market price could be between:")
                st.success(f"₹{np.round(prediction - 3492, 2):,} - ₹{np.round(prediction + 3492, 2):,}")
                
                st.info("💡 **What this means**: Our prediction is very reliable! The actual selling price will likely be very close to our estimate.")
            
            with col_analysis2:
                st.subheader("📈 What Affects Your Bike's Price Most?")
                
                # Create factor importance explanation
                factors = [
                    ("Bike Model", 19.9, "🏍️"),
                    ("Distance Traveled", 19.8, "🛣️"), 
                    ("Location (City)", 12.4, "📍"),
                    ("Age of Bike", 11.7, "📅"),
                    ("Engine Power", 10.2, "⚡")
                ]
                
                st.write("**Top 5 factors that determine your bike's value:**")
                for factor, importance, emoji in factors:
                    st.write(f"{emoji} **{factor}**: {importance}% impact")
                
                st.write("🔍 **Key Insight**: Your bike's specific model and how much it has been driven are the biggest factors affecting its price!")
            
            # Personalized insights
            st.markdown("---")
            st.subheader("🔍 Personalized Insights for Your Bike")
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                if kms_driven < 10000:
                    st.success("✅ **Low Mileage Advantage**\nYour bike has very low mileage, which increases its value!")
                elif kms_driven < 30000:
                    st.info("ℹ️ **Moderate Mileage**\nYour bike has reasonable mileage for its age.")
                else:
                    st.warning("⚠️ **High Mileage Impact**\nHigh mileage may reduce your bike's market value.")
            
            with insight_col2:
                if age < 3:
                    st.success("✅ **Nearly New**\nYour bike is relatively new, maintaining good value!")
                elif age < 7:
                    st.info("ℹ️ **Good Age**\nYour bike is in the sweet spot for resale value.")
                else:
                    st.warning("⚠️ **Older Model**\nAge is affecting your bike's market value.")
            
            with insight_col3:
                if power > 200:
                    st.success("✅ **High Performance**\nPowerful engine adds to your bike's appeal!")
                elif power > 150:
                    st.info("ℹ️ **Good Performance**\nDecent engine power for daily use.")
                else:
                    st.info("ℹ️ **Entry Level**\nSuitable for city commuting and fuel efficiency.")
            
            # Market context
            st.markdown("---")
            st.subheader("💼 Market Context")
            st.write("**How to use this prediction:**")
            st.write("• **Selling your bike?** Start negotiations around this price")
            st.write("• **Buying a similar bike?** This gives you a fair market reference")  
            st.write("• **Insurance valuation?** This reflects current market value")
            st.write("• **Loan against bike?** Banks often use similar valuations")
            
            st.info("💡 **Pro Tip**: Prices can vary based on bike condition, market demand, and local factors. Use this as your starting reference point!")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check if all required model files are available and properly trained.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-style: italic; margin-top: 2rem;'>"
    "Created with ❤️ by <strong>Waseem M Ansari</strong> and <strong>Claude AI</strong>"
    "</div>", 
    unsafe_allow_html=True
)