
import os
import joblib
import numpy as np
import json
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go

# ── LOAD MODELS ──
xgb    = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
U      = np.load('svd_U.npy')
sigma  = np.load('svd_sigma.npy')
Vt     = np.load('svd_Vt.npy')

with open('user_id_to_idx.json') as f:
    user_id_to_idx = {int(k):int(v) for k,v in json.load(f).items()}
with open('idx_to_cluster.json') as f:
    idx_to_cluster = {int(k):int(v) for k,v in json.load(f).items()}
with open('config.json') as f:
    config = json.load(f)

features      = config['features']
numerical_cols = config['numerical_cols']

sigma_diag       = np.diag(sigma)
predicted_scores = np.dot(np.dot(U, sigma_diag), Vt)

# ── STATIC DATA ──
BLUE   = "#003580"
YELLOW = "#FFC72C"
LIGHT  = "#0071C2"
PALE   = "#D6E4F7"
BG     = "#F4F6F9"
WHITE  = "#FFFFFF"
DARK   = "#0A0F1E"
MID    = "#6B7280"
BORDER = "#E4E4E7"
NAVY2  = "#001F4D"

lr_map5     = 0.1209
rf_map5     = 0.3936
lgbm_map5   = 0.3249
svd_map5    = 0.4850
xgb_map5    = 0.5323
hybrid_map5 = 0.6369
control_map5   = 0.0232
treatment_map5 = 0.6381

shap_labels = ["Geography","Local Market","Destination","Country",
               "Travel Distance","Planning Behaviour","Trip Type",
               "Group Size","User Origin","Booking Channel","Platform",
               "Family Travel","Package Deal","Region","Room Count",
               "Device Type","Family Flag"]
shap_vals   = [0.69,0.59,0.32,0.22,0.13,0.13,0.11,0.07,0.06,
               0.05,0.04,0.03,0.03,0.02,0.02,0.01,0.001]

monthly_bookings = [2842,2150,2748,2821,3124,3486,3952,4398,3621,3187,2943,3782]
mobile_vals  = [312,241,298,321,356,412,478,534,445,398,356,445]
desktop_vals = [2530,1909,2450,2500,2768,3074,3474,3864,3176,2789,2587,3337]

family_counts = [33440, 7614]
season_counts = {"Summer":13842,"Autumn":9821,"Spring":9634,"Winter":7757}

persona_data = {
    "Spontaneous Explorer":{"users":8084,"recency":23,"frequency":3.1,"monetary":2.1,"color":LIGHT,
        "points":["Books 2-3 weeks out. Target with urgency messaging and flash deals.",
                  "Largest segment at 73%. Highest volume marketing opportunity.",
                  "Short stays signal city-break preference. Promote weekend packages."]},
    "Luxury Long-Stay":{"users":1304,"recency":54,"frequency":1.9,"monetary":6.4,"color":"#C9922A",
        "points":["Stays 3x longer than average. Highest revenue per booking.",
                  "Books 54 days ahead. Premium curated newsletters convert well.",
                  "Low frequency but high value. Prioritise for loyalty programme design."]},
    "Frequent Business":{"users":417,"recency":28,"frequency":25.5,"monetary":2.3,"color":BLUE,
        "points":["Books 25x per user. Highest lifetime value of any segment.",
                  "Short stays in commercial destinations. City-centre cluster focus.",
                  "Weekly availability alerts and flexible cancellation drive retention."]},
    "Careful Planner":{"users":1212,"recency":147,"frequency":2.5,"monetary":3.1,"color":"#6f42c1",
        "points":["Plans 5 months ahead. Earliest campaign window of any segment.",
                  "Deliberate decision-makers who respond to detailed comparisons.",
                  "Early-bird pricing and itinerary planning tools drive conversion."]},
}

# ── MEDIAN INPUT FOR RECOMMENDER ──
X_train_medians = {
    "site_name":2,"posa_continent":3,"user_location_country":66,
    "is_mobile":0,"is_package":0,"channel":9,"srch_adults_cnt":2,
    "srch_children_cnt":0,"srch_rm_cnt":1,"srch_destination_id":8250,
    "hotel_continent":2,"hotel_country":50,"hotel_market":628,
    "stay_duration":0.0,"booking_lead_time":0.0,
    "is_family":0,"orig_destination_distance":0.0
}

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

def kpi(label, value, sub, accent=BLUE):
    return html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"20px 22px","position":"relative","overflow":"hidden"}, children=[
        html.Div(style={"position":"absolute","top":0,"left":0,"right":0,"height":"3px","background":accent}),
        html.P(label, style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 10px","fontWeight":"600"}),
        html.Div(value, style={"color":DARK,"fontSize":"32px","fontWeight":"800","letterSpacing":"-1px","lineHeight":"1","marginBottom":"6px","fontFamily":"Georgia,serif"}),
        html.P(sub, style={"color":MID,"fontSize":"11px","margin":0})
    ])

def strategy_card(title, points, accent):
    return html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"20px","position":"relative","overflow":"hidden"}, children=[
        html.Div(style={"position":"absolute","top":0,"left":0,"bottom":0,"width":"3px","background":accent}),
        html.P(title, style={"color":accent,"fontSize":"10px","fontWeight":"700","letterSpacing":"2px","textTransform":"uppercase","margin":"0 0 14px"}),
        html.Div([html.Div(style={"display":"flex","gap":"10px","marginBottom":"10px","alignItems":"flex-start"}, children=[
            html.Div(style={"width":"5px","height":"5px","borderRadius":"50%","background":accent,"marginTop":"6px","flexShrink":"0"}),
            html.P(p, style={"fontSize":"13px","color":DARK,"margin":0,"lineHeight":"1.5","fontWeight":"400"})
        ]) for p in points])
    ])

app.layout = html.Div(style={"backgroundColor":BG,"fontFamily":"\'Segoe UI\',system-ui,sans-serif","minHeight":"100vh"}, children=[
    html.Div(style={"background":f"linear-gradient(135deg,{NAVY2} 0%,{BLUE} 100%)","padding":"0 40px","display":"flex","justifyContent":"space-between","alignItems":"center","height":"56px"}, children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"12px"}, children=[
            html.Div(style={"width":"30px","height":"30px","borderRadius":"8px","background":YELLOW,"display":"flex","alignItems":"center","justifyContent":"center","fontWeight":"800","fontSize":"14px","color":BLUE}, children="E"),
            html.Div([
                html.Span("Expedia Intelligence Suite", style={"color":WHITE,"fontWeight":"700","fontSize":"14px"}),
                html.Span("  |  Hotel Recommendation Platform", style={"color":"rgba(255,255,255,0.45)","fontSize":"12px"})
            ])
        ]),
        html.Div(style={"display":"flex","alignItems":"center","gap":"7px"}, children=[
            html.Div(style={"width":"7px","height":"7px","borderRadius":"50%","background":"#4ADE80"}),
            html.Span("Hybrid Model Active", style={"color":"#4ADE80","fontSize":"11px","fontWeight":"600"})
        ])
    ]),

    dcc.Tabs(id="tabs", value="tab1", style={"backgroundColor":WHITE,"borderBottom":f"1px solid {BORDER}","paddingLeft":"32px"}, children=[
        dcc.Tab(label="Executive Summary",    value="tab1", style={"padding":"12px 18px","fontSize":"12px","fontWeight":"500","color":MID,"border":"none","backgroundColor":WHITE}, selected_style={"padding":"12px 18px","fontSize":"12px","fontWeight":"700","color":BLUE,"borderBottom":f"2px solid {YELLOW}","backgroundColor":WHITE,"border":"none"}),
        dcc.Tab(label="Model Architecture",   value="tab2", style={"padding":"12px 18px","fontSize":"12px","fontWeight":"500","color":MID,"border":"none","backgroundColor":WHITE}, selected_style={"padding":"12px 18px","fontSize":"12px","fontWeight":"700","color":BLUE,"borderBottom":f"2px solid {YELLOW}","backgroundColor":WHITE,"border":"none"}),
        dcc.Tab(label="What Drives Bookings", value="tab3", style={"padding":"12px 18px","fontSize":"12px","fontWeight":"500","color":MID,"border":"none","backgroundColor":WHITE}, selected_style={"padding":"12px 18px","fontSize":"12px","fontWeight":"700","color":BLUE,"borderBottom":f"2px solid {YELLOW}","backgroundColor":WHITE,"border":"none"}),
        dcc.Tab(label="Deployment Validation",value="tab4", style={"padding":"12px 18px","fontSize":"12px","fontWeight":"500","color":MID,"border":"none","backgroundColor":WHITE}, selected_style={"padding":"12px 18px","fontSize":"12px","fontWeight":"700","color":BLUE,"borderBottom":f"2px solid {YELLOW}","backgroundColor":WHITE,"border":"none"}),
        dcc.Tab(label="Customer Intelligence",value="tab5", style={"padding":"12px 18px","fontSize":"12px","fontWeight":"500","color":MID,"border":"none","backgroundColor":WHITE}, selected_style={"padding":"12px 18px","fontSize":"12px","fontWeight":"700","color":BLUE,"borderBottom":f"2px solid {YELLOW}","backgroundColor":WHITE,"border":"none"}),
        dcc.Tab(label="Live Recommender",     value="tab6", style={"padding":"12px 18px","fontSize":"12px","fontWeight":"500","color":MID,"border":"none","backgroundColor":WHITE}, selected_style={"padding":"12px 18px","fontSize":"12px","fontWeight":"700","color":BLUE,"borderBottom":f"2px solid {YELLOW}","backgroundColor":WHITE,"border":"none"}),
    ]),
    html.Div(id="tab-content", style={"padding":"28px 40px","maxWidth":"1400px"})
])

@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render(tab):

    if tab == "tab1":
        monthly_fig = go.Figure(go.Bar(
            x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            y=monthly_bookings,
            marker=dict(color=[YELLOW if m==8 else PALE for m in range(1,13)],line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y:,} bookings<extra></extra>"
        ))
        monthly_fig.update_layout(plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=220,margin=dict(t=16,b=32,l=40,r=16),yaxis=dict(gridcolor="#F0F2F5",tickfont=dict(size=10)),xaxis=dict(tickfont=dict(size=10)),bargap=0.3)

        mobile_fig = go.Figure()
        mobile_fig.add_trace(go.Scatter(x=list(range(1,13)),y=mobile_vals,name="Mobile",line=dict(color=YELLOW,width=2.5),fill="tozeroy",fillcolor="rgba(255,199,44,0.08)",hovertemplate="Month %{x}: %{y:,}<extra>Mobile</extra>"))
        mobile_fig.add_trace(go.Scatter(x=list(range(1,13)),y=desktop_vals,name="Desktop",line=dict(color=BLUE,width=2.5),fill="tozeroy",fillcolor="rgba(0,53,128,0.06)",hovertemplate="Month %{x}: %{y:,}<extra>Desktop</extra>"))
        mobile_fig.update_layout(plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=220,margin=dict(t=16,b=32,l=40,r=16),legend=dict(orientation="h",y=1.15,font=dict(size=11)),yaxis=dict(gridcolor="#F0F2F5",tickfont=dict(size=10)),xaxis=dict(tickfont=dict(size=10)))

        return html.Div([
            html.Div(style={"background":f"linear-gradient(135deg,{NAVY2} 0%,{BLUE} 60%,{LIGHT} 100%)","borderRadius":"16px","padding":"32px 36px","marginBottom":"24px","position":"relative","overflow":"hidden"}, children=[
                html.Div(style={"position":"absolute","top":"-50px","right":"-50px","width":"240px","height":"240px","borderRadius":"50%","background":"rgba(255,199,44,0.07)","border":"1px solid rgba(255,199,44,0.12)"}),
                html.Div(style={"position":"absolute","bottom":"-70px","right":"100px","width":"180px","height":"180px","borderRadius":"50%","background":"rgba(255,255,255,0.03)"}),
                html.Div(style={"position":"relative","zIndex":"2"}, children=[
                    html.Div(style={"display":"flex","alignItems":"center","gap":"8px","marginBottom":"14px"}, children=[
                        html.Div(style={"width":"6px","height":"6px","borderRadius":"50%","background":"#4ADE80"}),
                        html.Span("PRODUCTION READY",style={"color":"rgba(255,255,255,0.55)","fontSize":"10px","letterSpacing":"2px","fontWeight":"700"})
                    ]),
                    html.H2("Personalised recommendations drive conversion.",style={"color":WHITE,"fontSize":"22px","fontWeight":"800","margin":"0 0 10px","letterSpacing":"-0.5px","lineHeight":"1.2"}),
                    html.P("41,000+ real booking transactions. Six models evaluated. One production-ready recommendation engine delivering 26x better results than the random baseline.",style={"color":"rgba(255,255,255,0.65)","fontSize":"13px","margin":0,"lineHeight":"1.7","maxWidth":"640px"})
                ])
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"14px","marginBottom":"22px"}, children=[
                kpi("Recommendation Accuracy","63.7%","Hybrid Model top-5 hit rate",YELLOW),
                kpi("Conversion Uplift","26x","vs random baseline",LIGHT),
                kpi("Traveller Profiles","41,054","Real booking transactions",BLUE),
                kpi("Customer Segments","4","Recency-Frequency-Monetary personas","#7C3AED"),
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"14px","marginBottom":"22px"}, children=[
                strategy_card("PRODUCT STRATEGY",["Destination intelligence is the primary conversion lever.","Geography drives 3x more impact than any user attribute.","Mobile optimisation is secondary. Desktop drives 85% of volume."],BLUE),
                strategy_card("MARKETING STRATEGY",["Careful Planners convert best 5 months ahead of travel.","Spontaneous Explorers respond to urgency and flash deals.","Family segment stays 8% longer. Target with package bundles."],"#C9922A"),
                strategy_card("COMMERCIAL IMPACT",["63.7% top-5 accuracy translates to higher booking rates.","Business travellers (4% of users) drive outsized repeat revenue.","Validated at 99.99% confidence. Cleared for deployment."],LIGHT),
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px"}, children=[
                html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px"}, children=[
                    html.P("BOOKING VOLUME BY MONTH",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 2px","fontWeight":"600"}),
                    html.P("August peaks. Summer holiday demand drives highest conversion window.",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 8px"}),
                    dcc.Graph(figure=monthly_fig,config={"displayModeBar":False})
                ]),
                html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px"}, children=[
                    html.P("DEVICE SPLIT OVER TIME",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 2px","fontWeight":"600"}),
                    html.P("Desktop dominates at 85%. Mobile share grows steadily throughout the year.",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 8px"}),
                    dcc.Graph(figure=mobile_fig,config={"displayModeBar":False})
                ]),
            ])
        ])

    elif tab == "tab2":
        model_groups = {
            "Content-Based Filtering":{"desc":"Learns from traveller attributes to predict hotel preference. Destination, device, trip length, and group size all inform the recommendation.","color":LIGHT,"tag":"Feature-Driven","models":[
                {"name":"Logistic Regression","acc":lr_map5,"note":"Baseline linear model. Establishes minimum performance threshold."},
                {"name":"Random Forest","acc":rf_map5,"note":"Ensemble of 100 independent decision trees voted together."},
                {"name":"LightGBM","acc":lgbm_map5,"note":"Gradient boosting with leaf-wise growth. Fast, requires careful tuning."},
                {"name":"XGBoost","acc":xgb_map5,"note":"Sequential boosting. Each tree corrects the previous. Best content-based model."},
            ]},
            "Collaborative Filtering":{"desc":"Learns from behavioural history. Surfaces clusters that users with similar booking patterns have historically chosen.","color":"#6f42c1","tag":"Behaviour-Driven","models":[
                {"name":"SVD Matrix Factorisation","acc":svd_map5,"note":"Decomposes 11,017 user booking patterns into 50 latent preference dimensions."},
            ]}
        }
        group_cards = []
        for gname, gdata in model_groups.items():
            rows = []
            for m in gdata["models"]:
                pct = m["acc"]*100
                bar_w = min((pct/0.75)*100,100)
                rows.append(html.Div(style={"borderBottom":f"1px solid {BORDER}","paddingBottom":"14px","marginBottom":"14px"}, children=[
                    html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"flex-start","marginBottom":"8px"}, children=[
                        html.Div([html.P(m["name"],style={"fontWeight":"700","color":DARK,"margin":"0 0 2px","fontSize":"13px"}),html.P(m["note"],style={"color":MID,"fontSize":"11px","margin":0,"lineHeight":"1.5"})]),
                        html.Div(f"{pct:.1f}%",style={"color":gdata["color"],"fontWeight":"800","fontSize":"20px","letterSpacing":"-0.5px","fontFamily":"Georgia,serif","marginLeft":"16px","flexShrink":"0"})
                    ]),
                    html.Div(style={"background":"#F0F2F5","borderRadius":"4px","height":"6px","overflow":"hidden"}, children=[
                        html.Div(style={"background":gdata["color"],"height":"100%","borderRadius":"4px","width":f"{bar_w:.0f}%"})
                    ])
                ]))
            group_cards.append(html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"22px"}, children=[
                html.Div(style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"6px"}, children=[
                    html.Span(gdata["tag"],style={"background":gdata["color"]+"18","color":gdata["color"],"fontSize":"10px","fontWeight":"700","letterSpacing":"1px","padding":"3px 10px","borderRadius":"100px","border":f"1px solid {gdata['color']}35"}),
                    html.H3(gname,style={"color":DARK,"margin":0,"fontSize":"15px","fontWeight":"700"})
                ]),
                html.P(gdata["desc"],style={"color":MID,"fontSize":"12px","marginBottom":"18px","lineHeight":"1.6"}),
                *rows
            ]))

        hybrid_card = html.Div(style={"background":f"linear-gradient(135deg,{NAVY2} 0%,{BLUE} 100%)","borderRadius":"14px","padding":"26px","marginBottom":"16px","border":"1px solid rgba(255,199,44,0.2)"}, children=[
            html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center"}, children=[
                html.Div(style={"flex":1}, children=[
                    html.Div(style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"10px"}, children=[
                        html.Span("Best Performer",style={"background":"rgba(255,199,44,0.18)","color":YELLOW,"fontSize":"10px","fontWeight":"700","letterSpacing":"1px","padding":"3px 10px","borderRadius":"100px"}),
                        html.H3("Hybrid Model",style={"color":WHITE,"margin":0,"fontSize":"18px","fontWeight":"800"})
                    ]),
                    html.P("Two filtering paradigms evaluated across six models. Content-based learns from traveller features. Collaborative learns from booking behaviour. The hybrid captures what each misses independently.",style={"color":"rgba(255,255,255,0.72)","fontSize":"13px","lineHeight":"1.7","maxWidth":"560px","margin":"0 0 16px"}),
                    html.Div(style={"display":"flex","gap":"32px"}, children=[
                        html.Div([html.P("Content-Based Weight",style={"color":"rgba(255,255,255,0.45)","fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","margin":"0 0 3px"}),html.P("XGBoost — 70%",style={"color":YELLOW,"fontWeight":"700","fontSize":"13px","margin":0})]),
                        html.Div([html.P("Collaborative Weight",style={"color":"rgba(255,255,255,0.45)","fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","margin":"0 0 3px"}),html.P("SVD — 30%",style={"color":YELLOW,"fontWeight":"700","fontSize":"13px","margin":0})]),
                        html.Div([html.P("Lift over XGBoost",style={"color":"rgba(255,255,255,0.45)","fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","margin":"0 0 3px"}),html.P("+10.5 points",style={"color":YELLOW,"fontWeight":"700","fontSize":"13px","margin":0})]),
                    ])
                ]),
                html.Div(style={"background":"rgba(255,255,255,0.08)","borderRadius":"12px","padding":"22px 30px","textAlign":"center","marginLeft":"32px","flexShrink":"0"}, children=[
                    html.P("Final Accuracy",style={"color":"rgba(255,255,255,0.45)","fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","margin":"0 0 6px"}),
                    html.Div(f"{hybrid_map5*100:.1f}%",style={"color":YELLOW,"fontSize":"48px","fontWeight":"800","letterSpacing":"-2px","lineHeight":"1","fontFamily":"Georgia,serif"}),
                    html.P("MAP@5 Score",style={"color":"rgba(255,255,255,0.45)","fontSize":"10px","margin":"6px 0 0"})
                ])
            ])
        ])

        tree_connector = html.Div(style={"display":"flex","justifyContent":"center","alignItems":"center","padding":"6px 0"}, children=[
            html.Div(style={"flex":"1","height":"1px","background":f"linear-gradient(90deg,transparent,{BORDER})"}),
            html.Div(style={"display":"flex","alignItems":"center","gap":"8px","padding":"0 20px"}, children=[
                html.Div(style={"width":"8px","height":"8px","borderRadius":"50%","background":LIGHT}),
                html.Div(style={"width":"32px","height":"1px","background":BORDER}),
                html.Div(style={"width":"8px","height":"8px","borderRadius":"50%","background":YELLOW}),
                html.Div(style={"width":"32px","height":"1px","background":BORDER}),
                html.Div(style={"width":"8px","height":"8px","borderRadius":"50%","background":"#6f42c1"}),
            ]),
            html.Div(style={"flex":"1","height":"1px","background":f"linear-gradient(90deg,{BORDER},transparent)"}),
        ])

        return html.Div([
            html.H2("Model Architecture",style={"color":DARK,"fontSize":"20px","fontWeight":"800","letterSpacing":"-0.3px","margin":"0 0 4px"}),
            html.P("Two filtering paradigms. Content-based learns from features. Collaborative learns from behaviour. Hybrid combines the best of both.",style={"color":MID,"fontSize":"13px","margin":"0 0 20px"}),
            hybrid_card,
            tree_connector,
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px"}, children=group_cards),
        ])

    elif tab == "tab3":
        shap_fig = go.Figure(go.Bar(
            x=shap_vals[::-1],y=shap_labels[::-1],orientation="h",
            marker=dict(color=[YELLOW if v>0.3 else LIGHT if v>0.1 else PALE for v in shap_vals[::-1]],line=dict(width=0)),
            text=[f"{v:.2f}" for v in shap_vals[::-1]],textposition="outside",
            hovertemplate="<b>%{y}</b><br>Impact Score: %{x:.3f}<extra></extra>"
        ))
        shap_fig.update_layout(plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=420,margin=dict(l=150,r=70,t=8,b=8),xaxis=dict(gridcolor="#F0F2F5",tickfont=dict(size=10)),yaxis=dict(tickfont=dict(size=11)),bargap=0.3)

        family_fig = go.Figure(go.Pie(
            labels=["Solo / Couple","Family"],values=family_counts,
            marker=dict(colors=[BLUE,YELLOW],line=dict(color=WHITE,width=2)),
            hole=0.65,textinfo="percent",textfont=dict(size=12),
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"
        ))
        family_fig.update_layout(height=200,margin=dict(t=8,b=8,l=8,r=8),paper_bgcolor=WHITE,showlegend=True,legend=dict(orientation="h",y=-0.15,font=dict(size=11)))

        season_fig = go.Figure(go.Bar(
            x=list(season_counts.keys()),y=list(season_counts.values()),
            marker=dict(color=[YELLOW,LIGHT,BLUE,"#6f42c1"],line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y:,} bookings<extra></extra>"
        ))
        season_fig.update_layout(plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=200,margin=dict(t=8,b=32,l=40,r=8),yaxis=dict(gridcolor="#F0F2F5",tickfont=dict(size=10)),xaxis=dict(tickfont=dict(size=11)),bargap=0.35)

        def insight_card(title, points, accent):
            return html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"20px","position":"relative","overflow":"hidden"}, children=[
                html.Div(style={"position":"absolute","top":0,"left":0,"bottom":0,"width":"3px","background":accent}),
                html.P(title,style={"color":accent,"fontSize":"10px","fontWeight":"700","letterSpacing":"2px","textTransform":"uppercase","margin":"0 0 14px"}),
                html.Div([html.Div(style={"display":"flex","gap":"10px","marginBottom":"10px","alignItems":"flex-start"}, children=[
                    html.Div(style={"width":"5px","height":"5px","borderRadius":"50%","background":accent,"marginTop":"6px","flexShrink":"0"}),
                    html.P(p,style={"fontSize":"13px","color":DARK,"margin":0,"lineHeight":"1.6","fontWeight":"400"})
                ]) for p in points])
            ])

        return html.Div([
            html.H2("What Drives a Hotel Recommendation?",style={"color":DARK,"fontSize":"20px","fontWeight":"800","letterSpacing":"-0.3px","margin":"0 0 4px"}),
            html.P("SHAP values reveal which inputs carry the most weight in the model.",style={"color":MID,"fontSize":"13px","margin":"0 0 20px"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"14px","marginBottom":"20px"}, children=[
                insight_card("PRODUCT STRATEGY",["Destination data quality is the highest-leverage investment.","Geography drives 3x more impact than any user attribute.","Device-specific recommendation logic has minimal impact."],BLUE),
                insight_card("MARKETING STRATEGY",["Planning behaviour splits users into two distinct campaign audiences.","Trip type predicts budget tier. Longer stays signal premium preference.","Group size targeting: 1 adult signals business, 2 or more signals leisure."],"#C9922A"),
                insight_card("COMMERCIAL STRATEGY",["Package deals represent an untapped upsell opportunity.","Booking channel reveals which sources convert to highest-value clusters.","Long-haul travellers book materially different hotel tiers."],LIGHT),
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"2fr 1fr","gap":"14px"}, children=[
                html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px"}, children=[
                    html.P("FEATURE IMPACT RANKING",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 2px","fontWeight":"600"}),
                    html.P("Geography and destination outweigh all user attributes combined.",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 8px"}),
                    html.Div(style={"display":"flex","gap":"16px","marginBottom":"12px"}, children=[
                        html.Div(style={"display":"flex","alignItems":"center","gap":"6px"}, children=[html.Div(style={"width":"10px","height":"10px","borderRadius":"2px","background":YELLOW}),html.Span("High impact",style={"fontSize":"11px","color":MID})]),
                        html.Div(style={"display":"flex","alignItems":"center","gap":"6px"}, children=[html.Div(style={"width":"10px","height":"10px","borderRadius":"2px","background":LIGHT}),html.Span("Medium",style={"fontSize":"11px","color":MID})]),
                        html.Div(style={"display":"flex","alignItems":"center","gap":"6px"}, children=[html.Div(style={"width":"10px","height":"10px","borderRadius":"2px","background":PALE}),html.Span("Low",style={"fontSize":"11px","color":MID})]),
                    ]),
                    dcc.Graph(figure=shap_fig,config={"displayModeBar":False})
                ]),
                html.Div(style={"display":"flex","flexDirection":"column","gap":"14px"}, children=[
                    html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px","flex":"1"}, children=[
                        html.P("TRAVELLER TYPE",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 2px","fontWeight":"600"}),
                        html.P("81% solo or couple travel.",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 8px"}),
                        dcc.Graph(figure=family_fig,config={"displayModeBar":False})
                    ]),
                    html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px","flex":"1"}, children=[
                        html.P("BOOKINGS BY SEASON",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 2px","fontWeight":"600"}),
                        html.P("Summer drives peak demand.",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 8px"}),
                        dcc.Graph(figure=season_fig,config={"displayModeBar":False})
                    ]),
                ])
            ])
        ])

    elif tab == "tab4":
        ab_fig = go.Figure()
        ab_fig.add_trace(go.Bar(name="Control — Random Baseline",x=["Control"],y=[control_map5],marker=dict(color=PALE,line=dict(width=0)),text=[f"{control_map5:.1%}"],textposition="outside",hovertemplate="Control: %{y:.4f}<extra></extra>"))
        ab_fig.add_trace(go.Bar(name="Treatment — Hybrid Model",x=["Treatment"],y=[treatment_map5],marker=dict(color=YELLOW,line=dict(width=0)),text=[f"{treatment_map5:.1%}"],textposition="outside",hovertemplate="Treatment: %{y:.4f}<extra></extra>"))
        ab_fig.update_layout(barmode="group",plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=260,margin=dict(t=30,b=8,l=40,r=40),yaxis=dict(range=[0,0.75],tickformat=".0%",gridcolor="#F0F2F5"),legend=dict(orientation="h",y=1.15,font=dict(size=11)),bargap=0.4)

        return html.Div([
            html.Div(style={"background":f"linear-gradient(135deg,{NAVY2},{BLUE})","borderRadius":"14px","padding":"22px 28px","marginBottom":"22px","display":"flex","justifyContent":"space-between","alignItems":"center"}, children=[
                html.Div([
                    html.Div(style={"display":"flex","alignItems":"center","gap":"8px","marginBottom":"8px"}, children=[html.Div(style={"width":"7px","height":"7px","borderRadius":"50%","background":YELLOW}),html.Span("DEPLOYMENT DECISION",style={"color":"rgba(255,255,255,0.5)","fontSize":"10px","fontWeight":"700","letterSpacing":"2px"})]),
                    html.H2("Cleared for Production.",style={"color":WHITE,"margin":"0 0 8px","fontSize":"22px","fontWeight":"800","letterSpacing":"-0.5px"}),
                    html.P("Rigorous validation across 41,000 users confirms a 26x improvement over the current random baseline. At 99.99% statistical confidence, the evidence for immediate deployment is conclusive.",style={"color":"rgba(255,255,255,0.65)","margin":0,"fontSize":"13px","lineHeight":"1.7","maxWidth":"660px"})
                ]),
                html.Div(style={"background":"rgba(255,199,44,0.15)","borderRadius":"12px","padding":"18px 26px","textAlign":"center","marginLeft":"24px","flexShrink":"0","border":"1px solid rgba(255,199,44,0.25)"}, children=[
                    html.Div("26x",style={"color":YELLOW,"fontSize":"44px","fontWeight":"800","letterSpacing":"-2px","lineHeight":"1","fontFamily":"Georgia,serif"}),
                    html.P("Conversion Uplift",style={"color":"rgba(255,255,255,0.5)","fontSize":"10px","margin":"6px 0 0","textTransform":"uppercase","letterSpacing":"1px"})
                ])
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"14px","marginBottom":"22px"}, children=[
                kpi("Conversion Uplift","26x","vs random baseline",YELLOW),
                kpi("Statistical Confidence","99.99%","p-value < 0.0001",BLUE),
                kpi("Cohen's d Effect Size","2.15","Large — threshold is 0.8",LIGHT),
                kpi("95% Confidence Interval","63-64%","Treatment group range",BLUE),
            ]),
            html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px","marginBottom":"14px"}, children=[
                html.P("TEST METHODOLOGY",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 12px","fontWeight":"600"}),
                html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"16px"}, children=[
                    html.Div([html.P("Test Design",style={"color":MID,"fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","margin":"0 0 3px"}),html.P("Random 50/50 user split",style={"color":DARK,"fontSize":"12px","fontWeight":"600","margin":0})]),
                    html.Div([html.P("Statistical Test",style={"color":MID,"fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","margin":"0 0 3px"}),html.P("Independent samples t-test",style={"color":DARK,"fontSize":"12px","fontWeight":"600","margin":0})]),
                    html.Div([html.P("Effect Size Method",style={"color":MID,"fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","margin":"0 0 3px"}),html.P("Cohen's d",style={"color":DARK,"fontSize":"12px","fontWeight":"600","margin":0})]),
                    html.Div([html.P("Interval Method",style={"color":MID,"fontSize":"10px","textTransform":"uppercase","letterSpacing":"1px","margin":"0 0 3px"}),html.P("t-distribution, 95% CI",style={"color":DARK,"fontSize":"12px","fontWeight":"600","margin":0})]),
                ])
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px"}, children=[
                html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px"}, children=[
                    html.P("A/B TEST — ACCURACY COMPARISON",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 2px","fontWeight":"600"}),
                    html.P("Treatment outperforms control by 2,600%.",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 8px"}),
                    dcc.Graph(figure=ab_fig,config={"displayModeBar":False}),
                    html.P("The hybrid model finds the right hotel cluster in 63.8% of searches. The random baseline achieves 2.3%. This gap represents the direct business value of deploying the recommendation engine.",style={"color":MID,"fontSize":"12px","margin":"10px 0 0","lineHeight":"1.6"})
                ]),
                html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px"}, children=[
                    html.P("STATISTICAL VALIDATION",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 12px","fontWeight":"600"}),
                    *[html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center","padding":"12px 0","borderBottom":f"1px solid {BORDER}"}, children=[
                        html.P(label,style={"color":MID,"fontSize":"12px","margin":0}),
                        html.P(value,style={"color":DARK,"fontSize":"13px","fontWeight":"700","margin":0})
                    ]) for label,value in [
                        ("Control MAP@5","0.0232"),("Treatment MAP@5","0.6381"),
                        ("Uplift","+2,653%"),("p-value","< 0.0001"),
                        ("Cohen's d","2.15 (Large)"),("95% CI","(0.6328, 0.6434)"),
                        ("Control users","20,524"),("Treatment users","20,530"),
                    ]]
                ]),
            ]),
        ])

    elif tab == "tab5":
        recency_fig = go.Figure(go.Bar(
            x=[d["recency"] for d in persona_data.values()],
            y=list(persona_data.keys()),orientation="h",
            marker=dict(color=[d["color"] for d in persona_data.values()],line=dict(width=0)),
            text=[f"{d['recency']} days" for d in persona_data.values()],textposition="outside",
            hovertemplate="%{y}: %{x} days avg lead time<extra></extra>"
        ))
        recency_fig.update_layout(plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=200,margin=dict(t=8,b=8,l=150,r=80),xaxis=dict(gridcolor="#F0F2F5",tickfont=dict(size=10),range=[0,210]),yaxis=dict(tickfont=dict(size=10)),bargap=0.35)

        freq_donut = go.Figure(go.Pie(
            labels=list(persona_data.keys()),
            values=[d["frequency"]*d["users"] for d in persona_data.values()],
            marker=dict(colors=[d["color"] for d in persona_data.values()],line=dict(color=WHITE,width=2)),
            hole=0.6,textinfo="percent",textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>Share of total bookings: %{percent}<extra></extra>"
        ))
        freq_donut.update_layout(paper_bgcolor=WHITE,height=200,margin=dict(t=8,b=8,l=8,r=8),showlegend=False,
            annotations=[dict(text="Booking<br>Share",x=0.5,y=0.5,font_size=10,showarrow=False,font_color=MID)])

        monetary_fig = go.Figure(go.Bar(
            x=list(persona_data.keys()),
            y=[d["monetary"] for d in persona_data.values()],
            marker=dict(color=[d["color"] for d in persona_data.values()],opacity=[0.5,1.0,0.55,0.7],line=dict(width=0)),
            text=[f"{d['monetary']}n" for d in persona_data.values()],textposition="outside",
            hovertemplate="%{x}: %{y} avg nights<extra></extra>"
        ))
        monetary_fig.update_layout(
            plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=200,margin=dict(t=32,b=8,l=40,r=8),
            yaxis=dict(gridcolor="#F0F2F5",tickfont=dict(size=10),range=[0,9]),
            xaxis=dict(tickfont=dict(size=9)),bargap=0.4,
            shapes=[dict(type="line",x0=-0.5,x1=3.5,y0=2.7,y1=2.7,line=dict(color=MID,width=1.5,dash="dot"))],
            annotations=[dict(x=3.3,y=3.0,text="avg 2.7n",showarrow=False,font=dict(size=9,color=MID))]
        )

        rfm_fig = go.Figure()
        for name, data in persona_data.items():
            rfm_fig.add_trace(go.Scatter(
                x=[data["recency"]],y=[data["monetary"]],mode="markers+text",
                marker=dict(size=max(data["users"]/80,24),color=data["color"],opacity=0.85,line=dict(width=2,color=WHITE)),
                text=[name],textposition="top center",textfont=dict(size=11,color=DARK),name=name,
                hovertemplate=f"<b>{name}</b><br>Lead Time: {data['recency']} days<br>Avg Stay: {data['monetary']} nights<br>Users: {data['users']:,}<extra></extra>"
            ))
        rfm_fig.update_layout(
            plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=280,margin=dict(t=50,b=50,l=60,r=40),
            xaxis=dict(title="Booking Lead Time — Recency (days)",gridcolor="#F0F2F5",tickfont=dict(size=10),range=[-15,185]),
            yaxis=dict(title="Avg Stay — Monetary (nights)",gridcolor="#F0F2F5",tickfont=dict(size=10),range=[0,9]),
            legend=dict(orientation="h",y=-0.3,font=dict(size=10)),showlegend=True
        )

        persona_cards = []
        for name, data in persona_data.items():
            persona_cards.append(html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"20px","position":"relative","overflow":"hidden"}, children=[
                html.Div(style={"position":"absolute","top":0,"left":0,"right":0,"height":"3px","background":data["color"]}),
                html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"flex-start","marginBottom":"14px"}, children=[
                    html.H4(name,style={"color":DARK,"margin":0,"fontSize":"14px","fontWeight":"700"}),
                    html.Span(f"{data['users']:,}",style={"background":data["color"]+"18","color":data["color"],"fontSize":"11px","fontWeight":"700","padding":"3px 10px","borderRadius":"100px","border":f"1px solid {data['color']}30"})
                ]),
                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"8px","marginBottom":"14px"}, children=[
                    html.Div(style={"background":BG,"padding":"8px 10px","borderRadius":"8px"}, children=[html.P("Avg Stay",style={"color":MID,"fontSize":"9px","textTransform":"uppercase","letterSpacing":"1px","margin":"0 0 1px"}),html.P(f"{data['monetary']}n",style={"fontWeight":"800","fontSize":"16px","color":data["color"],"margin":0,"fontFamily":"Georgia,serif"})]),
                    html.Div(style={"background":BG,"padding":"8px 10px","borderRadius":"8px"}, children=[html.P("Lead Time",style={"color":MID,"fontSize":"9px","textTransform":"uppercase","letterSpacing":"1px","margin":"0 0 1px"}),html.P(f"{data['recency']}d",style={"fontWeight":"800","fontSize":"16px","color":data["color"],"margin":0,"fontFamily":"Georgia,serif"})]),
                ]),
                html.Div(style={"borderTop":f"1px solid {BORDER}","paddingTop":"12px"}, children=[
                    html.P("STRATEGIC ACTIONS",style={"color":MID,"fontSize":"9px","textTransform":"uppercase","letterSpacing":"1.5px","margin":"0 0 8px","fontWeight":"600"}),
                    html.Div([html.Div(style={"display":"flex","gap":"8px","marginBottom":"7px","alignItems":"flex-start"}, children=[
                        html.Div(style={"width":"4px","height":"4px","borderRadius":"50%","background":data["color"],"marginTop":"5px","flexShrink":"0"}),
                        html.P(p,style={"fontSize":"11px","color":DARK,"margin":0,"lineHeight":"1.6","fontWeight":"400"})
                    ]) for p in data["points"]])
                ])
            ]))

        return html.Div([
            html.Div(style={"background":f"linear-gradient(135deg,{NAVY2},{BLUE})","borderRadius":"14px","padding":"20px 28px","marginBottom":"22px","color":WHITE}, children=[
                html.H3("Four distinct traveller personas. One personalised platform.",style={"margin":"0 0 8px","fontSize":"18px","fontWeight":"800","letterSpacing":"-0.3px"}),
                html.P("Recency-Frequency-Monetary analysis across 11,017 unique users reveals four behavioural archetypes.",style={"margin":0,"fontSize":"13px","color":"rgba(255,255,255,0.7)","lineHeight":"1.6"})
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"14px","marginBottom":"20px"}, children=persona_cards),
            html.Div(style={"marginBottom":"14px"}, children=[
                html.P("RECENCY · FREQUENCY · MONETARY BREAKDOWN",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 4px","fontWeight":"600"}),
                html.P("Three dimensions that define each traveller persona.",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 14px"}),
                html.Div(style={"display":"grid","gridTemplateColumns":"1.2fr 0.8fr 1fr","gap":"14px"}, children=[
                    html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px"}, children=[
                        html.P("R — RECENCY",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 2px","fontWeight":"600"}),
                        html.P("How far ahead does each persona plan?",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 8px"}),
                        dcc.Graph(figure=recency_fig,config={"displayModeBar":False})
                    ]),
                    html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px"}, children=[
                        html.P("F — FREQUENCY",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 2px","fontWeight":"600"}),
                        html.P("Share of total booking volume.",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 8px"}),
                        dcc.Graph(figure=freq_donut,config={"displayModeBar":False})
                    ]),
                    html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px"}, children=[
                        html.P("M — MONETARY",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 2px","fontWeight":"600"}),
                        html.P("Luxury Long-Stay spends 3x longer than average.",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 8px"}),
                        dcc.Graph(figure=monetary_fig,config={"displayModeBar":False})
                    ]),
                ])
            ]),
            html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"16px 20px"}, children=[
                html.P("SEGMENT POSITIONING MAP",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 2px","fontWeight":"600"}),
                html.P("Recency vs Monetary. Bubble size proportional to number of users in each segment.",style={"color":DARK,"fontSize":"13px","fontWeight":"600","margin":"0 0 8px"}),
                dcc.Graph(figure=rfm_fig,config={"displayModeBar":False})
            ])
        ])

    elif tab == "tab6":
        return html.Div([
            html.H2("Live Recommendation Engine",style={"color":DARK,"fontSize":"20px","fontWeight":"800","letterSpacing":"-0.3px","margin":"0 0 4px"}),
            html.P("Configure a traveller search and generate personalised hotel cluster recommendations.",style={"color":MID,"fontSize":"13px","margin":"0 0 20px"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"360px 1fr","gap":"20px"}, children=[
                html.Div(style={"background":WHITE,"borderRadius":"14px","padding":"24px","border":f"1px solid {BORDER}"}, children=[
                    html.P("SEARCH PARAMETERS",style={"color":MID,"fontSize":"10px","letterSpacing":"1.5px","textTransform":"uppercase","margin":"0 0 18px","fontWeight":"600"}),
                    *[html.Div(style={"marginBottom":"18px"}, children=[
                        html.Label(lbl,style={"fontSize":"10px","color":MID,"textTransform":"uppercase","letterSpacing":"1px","display":"block","marginBottom":"8px","fontWeight":"600"}),
                        comp
                    ]) for lbl,comp in [
                        ("Destination ID",dcc.Input(id="dest-id",type="number",value=8250,style={"width":"100%","padding":"10px 14px","border":f"1px solid {BORDER}","borderRadius":"10px","fontSize":"14px","color":DARK,"outline":"none"})),
                        ("Adults",dcc.Slider(id="adults",min=1,max=6,step=1,value=2,marks={i:str(i) for i in range(1,7)},tooltip={"placement":"bottom","always_visible":True})),
                        ("Children",dcc.Slider(id="children",min=0,max=4,step=1,value=0,marks={i:str(i) for i in range(5)},tooltip={"placement":"bottom","always_visible":True})),
                        ("Stay Duration (nights)",dcc.Slider(id="stay",min=1,max=14,step=1,value=3,marks={1:"1",7:"7",14:"14"},tooltip={"placement":"bottom","always_visible":True})),
                        ("Booking Lead Time (days)",dcc.Slider(id="lead",min=0,max=180,step=5,value=30,marks={0:"0",90:"90",180:"180"},tooltip={"placement":"bottom","always_visible":True})),
                        ("Device",dcc.RadioItems(id="device",options=[{"label":"  Desktop","value":0},{"label":"  Mobile","value":1}],value=0,inline=True,style={"fontSize":"13px","color":DARK})),
                    ]],
                    html.Button("Generate Recommendations",id="recommend-btn",style={"width":"100%","backgroundColor":BLUE,"color":WHITE,"border":"none","padding":"13px","fontSize":"12px","fontWeight":"700","borderRadius":"10px","cursor":"pointer","letterSpacing":"1px","marginTop":"4px"})
                ]),
                html.Div(id="recommendation-output",children=[
                    html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"48px","textAlign":"center","display":"flex","alignItems":"center","justifyContent":"center","flexDirection":"column","minHeight":"400px"}, children=[
                        html.Div(style={"width":"48px","height":"48px","borderRadius":"12px","background":PALE,"margin":"0 auto 16px","display":"flex","alignItems":"center","justifyContent":"center"}, children=[html.Div(style={"width":"20px","height":"20px","borderRadius":"50%","border":f"2px solid {LIGHT}"})]),
                        html.P("Configure search parameters and click Generate",style={"color":MID,"fontSize":"14px","margin":0,"fontWeight":"500"})
                    ])
                ])
            ])
        ])

@app.callback(
    Output("recommendation-output","children"),
    Input("recommend-btn","n_clicks"),
    [Input("dest-id","value"),Input("adults","value"),Input("children","value"),Input("stay","value"),Input("lead","value"),Input("device","value")],
    prevent_initial_call=True
)
def recommend(n_clicks,dest_id,adults,children,stay,lead,device):
    if not n_clicks: return ""
    sample = X_train_medians.copy()
    sample.update({"srch_destination_id":dest_id or 8250,"srch_adults_cnt":adults,"srch_children_cnt":children,"stay_duration":stay,"booking_lead_time":lead,"is_mobile":device,"is_family":1 if children>0 else 0})
    X_input = pd.DataFrame([sample])[features]
    X_input[numerical_cols] = scaler.transform(X_input[numerical_cols])
    probs = xgb.predict_proba(X_input)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    top5_clusters = xgb.classes_[top5_idx]
    top5_probs = probs[top5_idx]

    if children>0:   persona,action,pc="Family Traveller","Promote family clusters and package bundles",LIGHT
    elif lead>60:    persona,action,pc="Careful Planner","Send early-bird pricing and destination guides","#6f42c1"
    elif adults==1:  persona,action,pc="Business Traveller","City-centre clusters with flexible cancellation",BLUE
    elif stay>5:     persona,action,pc="Luxury Long-Stay","Curate premium clusters with extended stay benefits","#C9922A"
    else:            persona,action,pc="Spontaneous Explorer","Target with availability alerts and flash deals",LIGHT

    max_prob = float(max(top5_probs))
    conf_fig = go.Figure(go.Bar(
        x=[f"Cluster {c}" for c in top5_clusters],y=top5_probs,
        marker=dict(color=[YELLOW if i==0 else PALE for i in range(5)],line=dict(width=0)),
        text=[f"{p:.1%}" for p in top5_probs],textposition="outside",
        hovertemplate="<b>%{x}</b><br>Confidence: %{y:.2%}<extra></extra>"
    ))
    conf_fig.update_layout(plot_bgcolor=WHITE,paper_bgcolor=WHITE,height=230,margin=dict(t=52,b=32,l=8,r=8),yaxis=dict(tickformat=".0%",gridcolor="#F0F2F5",tickfont=dict(size=10),range=[0,max_prob*1.4]),xaxis=dict(tickfont=dict(size=11)),bargap=0.4)

    rec_rows = []
    for rank,(cluster,prob) in enumerate(zip(top5_clusters,top5_probs),1):
        rec_rows.append(html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center","padding":"12px 16px","borderRadius":"10px","marginBottom":"6px","background":BLUE if rank==1 else WHITE,"border":f"1px solid {BORDER}" if rank>1 else "none"}, children=[
            html.Div(style={"display":"flex","alignItems":"center","gap":"12px"}, children=[
                html.Div(f"#{rank}",style={"width":"28px","height":"28px","borderRadius":"8px","background":YELLOW if rank==1 else PALE,"color":DARK,"fontSize":"11px","fontWeight":"700","display":"flex","alignItems":"center","justifyContent":"center"}),
                html.Div([html.P(f"Hotel Cluster {cluster}",style={"fontWeight":"700","fontSize":"14px","margin":0,"color":WHITE if rank==1 else DARK}),html.P("Top recommendation" if rank==1 else f"Alternative {rank-1}",style={"fontSize":"10px","margin":0,"color":"rgba(255,255,255,0.55)" if rank==1 else MID})])
            ]),
            html.P(f"{prob:.1%}",style={"fontWeight":"800","fontSize":"16px","margin":0,"color":YELLOW if rank==1 else LIGHT,"fontFamily":"Georgia,serif"})
        ]))

    return html.Div(style={"background":WHITE,"borderRadius":"14px","border":f"1px solid {BORDER}","padding":"22px"}, children=[
        html.Div(style={"background":pc,"borderRadius":"12px","padding":"16px 20px","marginBottom":"18px","display":"flex","justifyContent":"space-between","alignItems":"center"}, children=[
            html.Div([html.P("IDENTIFIED PERSONA",style={"color":"rgba(255,255,255,0.6)","fontSize":"9px","textTransform":"uppercase","letterSpacing":"2px","margin":"0 0 3px"}),html.H3(persona,style={"color":WHITE,"margin":0,"fontSize":"17px","fontWeight":"800"})]),
            html.Div(style={"textAlign":"right","maxWidth":"200px"}, children=[html.P("RECOMMENDED APPROACH",style={"color":"rgba(255,255,255,0.6)","fontSize":"9px","textTransform":"uppercase","letterSpacing":"2px","margin":"0 0 3px"}),html.P(action,style={"color":WHITE,"fontSize":"11px","margin":0,"fontWeight":"500","lineHeight":"1.5"})])
        ]),
        html.P("CONFIDENCE BY CLUSTER",style={"color":MID,"fontSize":"9px","textTransform":"uppercase","letterSpacing":"2px","margin":"0 0 4px","fontWeight":"600"}),
        dcc.Graph(figure=conf_fig,config={"displayModeBar":False}),
        html.P("TOP 5 RECOMMENDATIONS",style={"color":MID,"fontSize":"9px","textTransform":"uppercase","letterSpacing":"2px","margin":"14px 0 10px","fontWeight":"600"}),
        *rec_rows
    ])

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
