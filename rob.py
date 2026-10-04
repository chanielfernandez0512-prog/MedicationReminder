
# ===============================================================
# PART 1: THE LOGIC (saving data, doses, machine learning)
# ===============================================================
import json
import os
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from sklearn.linear_model import LogisticRegression

# Lets the normal Run button work: if the file was started with plain
# Python, restart it the correct way using Streamlit.
if not st.runtime.exists():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    sys.exit()

DATA_FILE = "medication_data.json"
TIMEZONE = "Asia/Manila"   # change this if you are in a different time zone
MIN_RECORDS = 10           # records needed before the model can learn


# ---------------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------------
def now():
    """Current date and time in the chosen time zone."""
    return datetime.now(ZoneInfo(TIMEZONE))


def to_24h(hour_12, minute, am_pm):
    """Turn 8, 5, 'PM' into '20:05' (stored format, easy to sort)."""
    parsed = datetime.strptime(f"{hour_12}:{minute:02d} {am_pm}", "%I:%M %p")
    return parsed.strftime("%H:%M")


def format_time(time_24):
    """Turn '20:05' into '08:05 PM' for display."""
    return datetime.strptime(time_24, "%H:%M").strftime("%I:%M %p")


# ---------------------------------------------------------------
# SAVING AND LOADING
# ---------------------------------------------------------------
def load_data():
    """Read medications and history from the JSON file."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as file:
                return json.load(file)
        except json.JSONDecodeError:
            pass   # damaged file: start fresh
    return {"medications": [], "history": []}


def save_data(data):
    with open(DATA_FILE, "w") as file:
        json.dump(data, file, indent=2)


# ---------------------------------------------------------------
# MEDICATIONS
# ---------------------------------------------------------------
def find_medication(data, name):
    for med in data["medications"]:
        if med["name"].lower() == name.lower():
            return med
    return None


def add_medication(data, name, dosage):
    """Returns (success, message)."""
    name = name.strip()
    if name == "":
        return False, "Please enter a medication name."
    if find_medication(data, name) is not None:
        return False, f"{name} is already in your list."

    data["medications"].append(
        {"name": name, "dosage": dosage.strip(), "time": None})
    save_data(data)
    return True, f"{name} added. Next, set its time."


def set_time(data, name, time_24):
    med = find_medication(data, name)
    med["time"] = time_24
    save_data(data)


# ---------------------------------------------------------------
# DOSES AND HISTORY
# ---------------------------------------------------------------
def mark_dose(data, name, status):
    """Record a dose as 'taken' or 'missed'."""
    med = find_medication(data, name)
    today = now()

    data["history"].append({
        "name": med["name"],
        "date": today.strftime("%Y-%m-%d"),
        "time": med["time"],
        "is_weekend": 1 if today.weekday() >= 5 else 0,
        "status": status,
        "sample": False,
    })
    save_data(data)


def todays_status(data, name):
    """Return 'taken', 'missed', or None if not marked yet today."""
    today = now().strftime("%Y-%m-%d")
    for record in reversed(data["history"]):
        if record["name"] == name and record["date"] == today:
            return record["status"]
    return None


def adherence_rate(history):
    """Percent of doses taken, or None if there is no history."""
    if not history:
        return None
    taken = sum(1 for h in history if h["status"] == "taken")
    return taken / len(history) * 100


# ---------------------------------------------------------------
# SAMPLE DATA (for demo only)
# ---------------------------------------------------------------
def load_sample_history(data):
    """Practice data so the model can learn before you have real history.
    Pattern: doses are missed more often late at night and on weekends."""
    if any(h.get("sample") for h in data["history"]):
        return   # already loaded

    random.seed(1)
    for _ in range(60):
        hour = random.choice([8, 12, 18, 21, 22])
        weekend = random.randint(0, 1)

        chance_missed = 0.1
        if hour >= 21:
            chance_missed += 0.3
        if weekend:
            chance_missed += 0.2

        status = "missed" if random.random() < chance_missed else "taken"
        data["history"].append({
            "name": "(sample)", "date": "", "time": f"{hour:02d}:00",
            "is_weekend": weekend, "status": status, "sample": True,
        })
    save_data(data)


def clear_sample_history(data):
    data["history"] = [h for h in data["history"] if not h.get("sample")]
    save_data(data)


# ---------------------------------------------------------------
# MACHINE LEARNING
# ---------------------------------------------------------------
def train_model(history):
    """Learn from past doses. Returns None if there isn't enough history."""
    if len(history) < MIN_RECORDS:
        return None
    if len({h["status"] for h in history}) < 2:
        return None   # needs both 'taken' and 'missed' examples

    # Inputs (X): hour of the dose and weekend or not
    X = [[int(h["time"][:2]), h["is_weekend"]] for h in history]
    # Answers (y): 1 = missed, 0 = taken
    y = [1 if h["status"] == "missed" else 0 for h in history]

    model = LogisticRegression()
    model.fit(X, y)
    return model


def next_reminder(time_24):
    """The next time this reminder will happen (today or tomorrow)."""
    current = now()
    hour, minute = map(int, time_24.split(":"))
    target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= current:
        target += timedelta(days=1)
    return target


def predict_miss_risk(model, time_24):
    """Chance (0-100) of missing the next reminder, and when it is."""
    target = next_reminder(time_24)
    is_weekend = 1 if target.weekday() >= 5 else 0
    risk = model.predict_proba([[target.hour, is_weekend]])[0][1] * 100
    return risk, target


def risk_by_hour(model):
    """Predicted chance (0-100) of missing a dose at every hour of the day,
    for weekdays and for weekends. Used to draw the chart."""
    rows = []
    for hour in range(24):
        weekday = model.predict_proba([[hour, 0]])[0][1] * 100
        weekend = model.predict_proba([[hour, 1]])[0][1] * 100
        rows.append({"Hour": hour,
                     "Weekday": round(weekday, 1),
                     "Weekend": round(weekend, 1)})
    return rows


# ===============================================================
# PART 2: THE SCREEN (Streamlit app)
# ===============================================================
st.set_page_config(page_title="Medication Reminder", page_icon="💊")
st.title("💊 Medication Reminder")

data = load_data()


# ---------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------
def flash(message):
    """Remember a message to show after the page reloads."""
    st.session_state["flash"] = message


def play_alert():
    """Beep three times and vibrate the phone (vibration works on Android
    browsers only). The browser only allows this after the user has tapped
    the page at least once."""
    sound_and_vibration = """
    <script>
      try { navigator.vibrate([400, 200, 400, 200, 400]); } catch (e) {}
      try {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        const ctx = new Ctx();
        if (ctx.state === "suspended") { ctx.resume(); }
        [0, 0.5, 1.0].forEach(function (start) {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.frequency.value = 880;
          gain.gain.value = 0.3;
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start(ctx.currentTime + start);
          osc.stop(ctx.currentTime + start + 0.3);
        });
      } catch (e) {}
    </script>
    """

    if hasattr(st, "iframe"):          # newer Streamlit
        st.iframe(sound_and_vibration, height=1)
    else:                              # older Streamlit
        import streamlit.components.v1 as components
        components.html(sound_and_vibration, height=0)


def scheduled_medications():
    """Medications that already have a time."""
    return [m for m in data["medications"] if m["time"]]


if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))


# ---------------------------------------------------------------
# SIDEBAR: DEMO DATA
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Demo data")
    st.caption("The prediction needs history to learn from. "
               "Load practice records to try it right away.")
    if st.button("Load sample history"):
        load_sample_history(data)
        flash("Sample history loaded.")
        st.rerun()
    if st.button("Clear sample history"):
        clear_sample_history(data)
        flash("Sample history cleared.")
        st.rerun()


# ---------------------------------------------------------------
# TABS
# ---------------------------------------------------------------
tab_remind, tab_add, tab_time, tab_mark, tab_history, tab_predict = st.tabs(
    ["🔔 Reminders", "➕ Add", "🕒 Set time", "✅ Mark dose",
     "📋 History", "🤖 Prediction"])


# ---- REMINDER NOTIFICATION (refreshes itself every 60 seconds) ----
@st.fragment(run_every=60)
def show_reminders():
    fresh = load_data()
    model = train_model(fresh["history"])
    current = now()

    st.caption(f"Current time: {current.strftime('%I:%M %p')}")
    st.caption("Tip: tap the page once and keep this screen open, "
               "so your phone allows the sound and vibration.")

    meds = [m for m in fresh["medications"] if m["time"]]
    if not meds:
        st.info("No reminders yet. Add a medication and set its time.")
        return

    notified = st.session_state.setdefault("notified", set())

    for med in sorted(meds, key=lambda m: m["time"]):
        label = (f"{med['name']} ({med['dosage']}) "
                 f"at {format_time(med['time'])}")
        status = todays_status(fresh, med["name"])

        if status == "taken":
            st.success(f"✅ {label} - taken")
        elif status == "missed":
            st.error(f"❌ {label} - missed")
        elif med["time"] <= current.strftime("%H:%M"):
            st.warning(f"🔔 TIME TO TAKE: {label}")
            key = f"{med['name']}-{current.strftime('%Y-%m-%d')}"
            if key not in notified:       # pop up only once per day
                st.toast(f"Time to take {med['name']}!", icon="🔔")
                play_alert()
                notified.add(key)
        else:
            text = f"⏰ Upcoming: {label}"
            if model is not None:
                risk, _ = predict_miss_risk(model, med["time"])
                text += f" | chance of missing: {risk:.0f}%"
            st.info(text)


with tab_remind:
    show_reminders()


# ---- ADD MEDICATION ----
with tab_add:
    with st.form("add_form", clear_on_submit=True):
        name = st.text_input("Medication name")
        dosage = st.text_input("Dosage (e.g. 500mg)")
        submitted = st.form_submit_button("Add medication")

    if submitted:
        ok, message = add_medication(data, name, dosage)
        if ok:
            flash(message)
            st.rerun()
        else:
            st.error(message)


# ---- SET TIME ----
with tab_time:
    names = [m["name"] for m in data["medications"]]
    if not names:
        st.info("Add a medication first.")
    else:
        chosen = st.selectbox("Medication", names, key="time_med")
        col1, col2, col3 = st.columns(3)
        hour = col1.selectbox("Hour", list(range(1, 13)))
        minute = col2.selectbox("Minute", list(range(60)),
                                format_func=lambda m: f"{m:02d}")
        am_pm = col3.selectbox("AM/PM", ["AM", "PM"])

        if st.button("Set time"):
            time_24 = to_24h(hour, minute, am_pm)
            set_time(data, chosen, time_24)
            flash(f"{chosen} is now scheduled at "
                  f"{format_time(time_24)}.")
            st.rerun()


# ---- MARK TAKEN / MISSED ----
with tab_mark:
    meds = scheduled_medications()
    if not meds:
        st.info("Set a time for at least one medication first.")
    else:
        chosen = st.selectbox("Medication", [m["name"] for m in meds],
                              key="mark_med")
        status = st.radio("Was the dose...", ["taken", "missed"],
                          horizontal=True)

        if st.button("Save"):
            mark_dose(data, chosen, status)
            flash(f"{chosen} marked as {status}.")
            st.rerun()


# ---- VIEW HISTORY ----
with tab_history:
    history = data["history"]
    if not history:
        st.info("No history yet.")
    else:
        rate = adherence_rate(history)
        col1, col2 = st.columns(2)
        col1.metric("Total records", len(history))
        col2.metric("Adherence rate", f"{rate:.0f}%")

        rows = [{
            "Date": h["date"] or "sample",
            "Medication": h["name"],
            "Time": format_time(h["time"]),
            "Status": h["status"],
        } for h in reversed(history)]
        st.dataframe(rows, width="stretch")


# ---- MACHINE LEARNING PREDICTION ----
with tab_predict:
    st.write("The app learns from your previous history to predict "
             "whether you are likely to miss your next reminder.")

    model = train_model(data["history"])
    meds = scheduled_medications()

    if model is None:
        st.info(f"Not enough history yet. The model needs at least "
                f"{MIN_RECORDS} records with both taken and missed "
                f"doses. Mark some doses, or load the sample history "
                f"from the sidebar.")
    elif not meds:
        st.info("Set a time for at least one medication first.")
    else:
        chosen = st.selectbox("Medication", [m["name"] for m in meds],
                              key="predict_med")
        med = find_medication(data, chosen)
        risk, target = predict_miss_risk(model, med["time"])

        st.metric("Chance of missing the next reminder", f"{risk:.0f}%")
        st.caption(f"Next reminder: {target.strftime('%A, %I:%M %p')} "
                   f"(learned from {len(data['history'])} records)")

        if risk >= 50:
            st.warning("Likely to miss: consider an extra reminder.")
        else:
            st.success("Unlikely to miss: a normal reminder is enough.")

        st.subheader("Chance of missing a dose, by hour")
        st.line_chart(
            risk_by_hour(model),
            x="Hour",
            y=["Weekday", "Weekend"],
            x_label="Hour of the day (0 = 12 AM, 12 = 12 PM, 23 = 11 PM)",
            y_label="Chance of missing (%)",
        )
        st.caption("Each line shows what the model learned from your history. "
                   "A higher line means doses at that hour are more "
                   "likely to be missed.")

    with st.expander("How does this work?"):
        st.write(
            "The model is a logistic regression (scikit-learn). It looks "
            "at two things from each past dose: the hour of the day and "
            "whether it was a weekend. It learns which combinations "
            "usually end in a missed dose, then gives a percentage for "
            "the next reminder.")
