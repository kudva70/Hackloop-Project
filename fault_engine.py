"""
AI Repair Assistant - Rule-Based Fault Diagnosis Engine
Grounded in deterministic, expert-curated decision rules for household & electronic devices:
  - Fans (Ceiling, Table, Exhaust)
  - Laptops / Notebooks
  - Bulbs (LED, Incandescent, Fluorescent)
  - Switchboards / Sockets / Circuit Breakers

Combines object detection results with user-reported symptoms to produce:
  1. Primary & Secondary Fault Identifications
  2. Supporting Evidence
  3. Safety Risk & Precaution Warnings
  4. Step-by-Step Troubleshooting / Verification
  5. Repair Complexity (Low / Medium / High / Dangerous)
  6. Final Actionable Recommendation (REPAIR DIY, SERVICE PRO, REPLACE)
"""

from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass, asdict


@dataclass
class FaultRule:
    fault_id: str
    title: str
    severity: str          # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    complexity: str        # "Low (Beginner)", "Medium (Intermediate)", "High (Advanced)", "Dangerous (Do Not DIY)"
    recommendation: str    # "REPAIR (DIY)", "SERVICE (Professional)", "REPLACE (New Device)"
    rationale: str
    required_tools: List[str]
    estimated_cost: str
    estimated_time: str
    safety_warnings: List[str]
    troubleshooting_steps: List[str]
    positive_keywords: List[str]
    negative_keywords: List[str]
    common_symptoms: List[str]
    co_occurrence_patterns: List[List[str]]  # High-confidence combos


# ---------------------------------------------------------------------------
# EXPERT KNOWLEDGE BASE
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: Dict[str, List[FaultRule]] = {
    # =======================================================================
    # 1. FAN (Ceiling / Table / Exhaust)
    # =======================================================================
    "fan": [
        FaultRule(
            fault_id="FAN_WEAK_CAPACITOR",
            title="Degraded or Blown Start/Run Capacitor",
            severity="MEDIUM",
            complexity="Medium (Intermediate)",
            recommendation="REPAIR (DIY)",
            rationale="A 2.5uF/3.15uF capacitor is inexpensive (₹50-₹120) and easily replaced with basic screwdrivers.",
            required_tools=["Insulated Screwdriver", "Wire Stripper", "Insulation Tape / Wire Nuts", "New Capacitor (2.5uF / 3.15uF 440V)"],
            estimated_cost="₹50 - ₹150 ($1 - $2)",
            estimated_time="20 - 30 minutes",
            safety_warnings=[
                "DANGER: Turn off the main electrical circuit breaker/isolator before touching any wires.",
                "Capacitors store residual electric charge even when turned off. Discharge by shorting terminals with an insulated-handle screwdriver before handling."
            ],
            troubleshooting_steps=[
                "1. Power OFF the fan switch and turn off the breaker.",
                "2. Check if blades spin smoothly by hand. If they spin freely without friction, the bearings are fine and the capacitor is the culprit.",
                "3. Remove the top decorative canopy to reveal the cylindrical capacitor.",
                "4. Inspect capacitor for bulging, leakage, or oily residue.",
                "5. Take a photo of wire connections for reference.",
                "6. Swap with a new capacitor of the EXACT same rating (usually 2.5uF for standard ceiling fans).",
                "7. Secure wire joints with wire nuts or heat-shrink tape and reassemble."
            ],
            positive_keywords=["hum", "humming", "slow", "slowly", "wont spin", "won't spin", "push to start", "spin by hand", "sluggish", "buzz", "buzzing", "capacitor"],
            negative_keywords=["jammed", "stuck", "smoke", "burnt smell", "burning"],
            common_symptoms=[
                "Fan hums or buzzes but blades do not spin",
                "Fan spins very slowly even at maximum regulator speed",
                "Blades only start spinning after giving them a push by hand"
            ],
            co_occurrence_patterns=[
                ["hum", "push"],
                ["slow", "regulator"],
                ["hum", "wont spin"],
                ["buzz", "slow"]
            ]
        ),
        FaultRule(
            fault_id="FAN_JAMMED_BEARINGS",
            title="Seized or Dry Bearings / Mechanical Friction",
            severity="MEDIUM",
            complexity="Medium (Intermediate)",
            recommendation="SERVICE (Professional) / REPAIR (DIY with lubrication)",
            rationale="Bearings have dried grease, rust, or grit. If light lubrication doesn't free it, bearing puller and replacement is needed.",
            required_tools=["Light Machine Oil (WD-40 / SAE 10W)", "Clean rag", "Screwdriver set"],
            estimated_cost="₹30 (oil) to ₹250 (new bearings)",
            estimated_time="30 - 45 minutes",
            safety_warnings=[
                "Disconnect electrical supply completely before working on the fan.",
                "Do not spray conductive aerosols inside the electric motor stator or wiring."
            ],
            troubleshooting_steps=[
                "1. Turn OFF power at the mains.",
                "2. Try rotating the fan blades manually by hand.",
                "3. If there is noticeable stiffness, grinding noise, or heavy resistance, the bearings are dry or seized.",
                "4. For table/pedestal fans: Apply 3-4 drops of sewing machine oil or synthetic motor oil to the rotor shaft bushings.",
                "5. Rotate by hand until motion frees up.",
                "6. For ceiling fans: Sealed ball bearings (6201/6202) may need replacement if grinding continues."
            ],
            positive_keywords=["stiff", "jammed", "hard to rotate", "resistance", "grinding", "squeak", "squeaking", "friction", "wont turn by hand", "noise", "rubbing"],
            negative_keywords=["loose", "wobble", "remote"],
            common_symptoms=[
                "Blades are stiff and hard to turn even by hand",
                "Loud squeaking, screeching, or grinding noise during rotation",
                "Motor gets hot quickly after a few minutes of trying to turn"
            ],
            co_occurrence_patterns=[
                ["stiff", "hand"],
                ["grinding", "noise"],
                ["squeak", "rotate"],
                ["jammed", "shaft"]
            ]
        ),
        FaultRule(
            fault_id="FAN_BURNT_STATOR_WINDING",
            title="Motor Stator Winding Burn / Short Circuit",
            severity="HIGH",
            complexity="High (Advanced)",
            recommendation="SERVICE (Professional) / REPLACE (New Motor)",
            rationale="Rewinding burnt copper stator coils requires specialized coil winding machinery, or full motor replacement.",
            required_tools=["Multimeter (Resistance/Continuity)", "Screwdrivers"],
            estimated_cost="₹350 - ₹600 (Rewinding) or ₹1,200 - ₹2,000 (New fan)",
            estimated_time="1 - 2 business days (Service Center)",
            safety_warnings=[
                "FIRE & SHOCK HAZARD: Do not leave a fan running if it emits burning smells or smoke.",
                "Burnt insulation can lead to short circuits and shock on the fan outer metal chassis."
            ],
            troubleshooting_steps=[
                "1. Immediately turn OFF power at the wall switch and circuit breaker.",
                "2. Disconnect fan from power supply.",
                "3. Smell around the motor housing for acrid burnt lacquer/copper odor.",
                "4. Measure continuity across the running and starting coils using a digital multimeter (Ohms mode).",
                "5. An open loop (infinite resistance) or direct chassis continuity indicates a burnt or shorted winding.",
                "6. Send stator to an electric motor rewinding specialist or replace the motor."
            ],
            positive_keywords=["smoke", "smell", "burning", "burnt", "spark", "black smoke", "acrid", "burnt copper", "shock"],
            negative_keywords=["wobble", "remote"],
            common_symptoms=[
                "Strong burnt plastic or burnt copper smell coming from motor housing",
                "Visible smoke or discoloration",
                "Fan completely dead with zero hum and zero movement"
            ],
            co_occurrence_patterns=[
                ["burning", "smell"],
                ["smoke", "motor"],
                ["burnt", "dead"]
            ]
        ),
        FaultRule(
            fault_id="FAN_BLADE_IMBALANCE_WOBBLE",
            title="Blade Misalignment or Loose Canopy Screws",
            severity="LOW",
            complexity="Low (Beginner)",
            recommendation="REPAIR (DIY)",
            rationale="Tightening blade holder screws or balancing blade weights takes 10 minutes with zero parts cost.",
            required_tools=["Cross-head / Phillips Screwdriver", "Ruler / Measuring Tape", "Fan Balancing Clip or Coin + Tape"],
            estimated_cost="₹0 ($0)",
            estimated_time="10 - 15 minutes",
            safety_warnings=[
                "Ensure fan is completely stopped and switched off before touching the blades.",
                "Inspect the downrod suspension pin and safety wire to prevent falling hazard."
            ],
            troubleshooting_steps=[
                "1. Switch OFF the fan and wait for it to come to a complete stop.",
                "2. Inspect each blade screw where it attaches to the motor housing. Tighten any loose screws firmly.",
                "3. Measure the vertical distance from the ceiling to the tip of each blade using a ruler.",
                "4. If one blade is higher or lower, gently flex the blade bracket (not the blade itself) until all tips match within 1/8 inch.",
                "5. Test the fan on medium speed to verify stability."
            ],
            positive_keywords=["wobble", "wobbly", "shaking", "vibrate", "vibrating", "rattling", "loose", "rattle", "screws", "swinging"],
            negative_keywords=["smoke", "burn"],
            common_symptoms=[
                "Fan wobbles or shakes vigorously while running",
                "Rattling or clicking noise from loose downrod or screws",
                "Blades visibly misaligned or bent"
            ],
            co_occurrence_patterns=[
                ["wobble", "speed"],
                ["shake", "noise"],
                ["rattle", "screws"]
            ]
        )
    ],

    # =======================================================================
    # 2. LAPTOP / NOTEBOOK
    # =======================================================================
    "laptop": [
        FaultRule(
            fault_id="LAPTOP_DEAD_BATTERY_CHARGER",
            title="Faulty Power Adapter / Dead Battery / DC Jack Failure",
            severity="MEDIUM",
            complexity="Low (Beginner)",
            recommendation="REPAIR (DIY / Replace Adapter)",
            rationale="Replacing an AC adapter or external/modular battery requires no internal teardown.",
            required_tools=["Compatible OEM Charger", "Voltmeter / Multimeter", "USB-C Power Delivery tester (if USB-C)"],
            estimated_cost="₹800 - ₹2,500 ($10 - $30)",
            estimated_time="10 minutes",
            safety_warnings=[
                "Never use generic unbranded cheap chargers; they can cause voltage spikes and battery thermal runaway.",
                "If battery is physically swollen (lifting trackpad/chassis), cease charging immediately. High fire risk."
            ],
            troubleshooting_steps=[
                "1. Perform a hard EC (Embedded Controller) reset: Unplug charger, hold power button down for 30 full seconds, then release.",
                "2. Inspect the charger cable for kinks, fraying, or loose pins in the charging barrel/USB-C connector.",
                "3. Try plugging into a different confirmed working wall socket without an extension strip.",
                "4. Check if the LED charging indicator illuminates when the charger is connected.",
                "5. If LED lights up but laptop won't turn on, test with a known good compatible charger."
            ],
            positive_keywords=["won't turn on", "wont turn on", "not turning on", "dead", "no power", "not charging", "charger", "plugged in not charging", "battery", "dies", "dc jack"],
            negative_keywords=["blue screen", "spill", "hinge"],
            common_symptoms=[
                "Laptop completely dead, no lights or fan activity when pressing power button",
                "Battery does not charge when plugged in, or drops to 0% immediately when unplugged",
                "Charging only works when wiggling the charger cable at a specific angle"
            ],
            co_occurrence_patterns=[
                ["not charging", "battery"],
                ["wont turn on", "charger"],
                ["dead", "plugged in"],
                ["power", "light"]
            ]
        ),
        FaultRule(
            fault_id="LAPTOP_OVERHEATING_DUST_THERMAL",
            title="Thermal Throttling / Clogged Fan / Dried Thermal Paste",
            severity="MEDIUM",
            complexity="Medium (Intermediate)",
            recommendation="REPAIR (DIY) / SERVICE (Professional)",
            rationale="Cleaning heatsink fins with compressed air and repasting thermal compound restores normal temps and performance.",
            required_tools=["Precision Screwdriver Set (PH00/T5)", "Compressed Air Can", "High-performance Thermal Paste (Arctic MX-4/Noctua)", "Isopropyl Alcohol (99%)"],
            estimated_cost="₹300 - ₹600 ($4 - $8)",
            estimated_time="30 - 45 minutes",
            safety_warnings=[
                "Always unplug the battery cable from the motherboard before touching the heatsink or fans.",
                "Use an anti-static wrist strap or touch a grounded metal object to prevent ESD damage."
            ],
            troubleshooting_steps=[
                "1. Check if the fan is spinning loudly even when idle or browsing simple websites.",
                "2. Feel the exhaust vents: if the fan is screaming but little air is coming out, the heatsink grill is clogged with dust.",
                "3. Power off, open the bottom panel with precision screwdrivers.",
                "4. Unplug the battery connector from the motherboard.",
                "5. Hold the fan blades gently (do not let them over-spin) and blow dust out with compressed air.",
                "6. If laptop is older than 2 years, unscrew heatsink, clean old crusty paste with 99% alcohol, apply a pea-sized dot of new paste, and remount."
            ],
            positive_keywords=["overheat", "overheating", "hot", "very hot", "fan loud", "fan noise", "shuts down randomly", "thermal throttling", "lagging", "high temp", "thermal paste"],
            negative_keywords=["liquid", "spill", "water", "cracked"],
            common_symptoms=[
                "Laptop gets extremely hot underneath, especially near the exhaust vent",
                "Cooling fan runs at maximum 100% speed continuously",
                "Laptop suddenly shuts down or restarts under heavy load (thermal protection)"
            ],
            co_occurrence_patterns=[
                ["hot", "fan"],
                ["overheating", "shuts down"],
                ["loud", "fan"],
                ["thermal", "throttling"]
            ]
        ),
        FaultRule(
            fault_id="LAPTOP_DISPLAY_RAM_ISSUE",
            title="Black Screen / RAM Contact Oxidation / Loose Display Ribbon Cable",
            severity="MEDIUM",
            complexity="Medium (Intermediate)",
            recommendation="REPAIR (DIY)",
            rationale="Reseating RAM sticks with an eraser or reconnecting the eDP display cable fixes 80% of 'power on with black screen' problems.",
            required_tools=["Precision Phillips/Torx screwdriver", "Pencil Eraser", "Microfiber cloth"],
            estimated_cost="₹0 ($0)",
            estimated_time="20 - 30 minutes",
            safety_warnings=[
                "Disconnect the battery before touching RAM slots or the display eDP cable. Shorting display pins can burn the motherboard backlight fuse!"
            ],
            troubleshooting_steps=[
                "1. Connect the laptop to an external TV or monitor via HDMI.",
                "2. If external monitor displays the desktop normally, the laptop motherboard/GPU is fine; the issue is the internal screen panel, eDP cable, or backlight fuse.",
                "3. If external monitor also stays black while power LEDs and fans turn on, RAM contact oxidation is the most common cause.",
                "4. Open bottom chassis, disconnect battery, unclip RAM modules.",
                "5. Rub the golden contact pins on both sides of the RAM stick with a clean pencil eraser to remove microscopic oxidation.",
                "6. Reinsert RAM firmly into the slots at a 30-degree angle and click down. Reconnect battery and power on."
            ],
            positive_keywords=["black screen", "blank screen", "no display", "screen dark", "keyboard lights up no display", "caps lock blinking", "ram", "display cable", "external monitor"],
            negative_keywords=["spill", "water", "hinge"],
            common_symptoms=[
                "Power LED and fan turn on, but the screen stays pitch black with no logo",
                "Caps Lock light blinks in a rhythmic code (e.g. 3 slow, 2 fast)",
                "Display only flickers or shows lines when moving the screen hinge"
            ],
            co_occurrence_patterns=[
                ["black screen", "power"],
                ["blank", "lights"],
                ["no display", "fan"],
                ["caps lock", "blink"]
            ]
        ),
        FaultRule(
            fault_id="LAPTOP_LIQUID_SPILL",
            title="Liquid Ingress / Liquid Corrosion Hazard",
            severity="CRITICAL",
            complexity="Dangerous (Do Not DIY)",
            recommendation="SERVICE (Professional Ultrasonic Cleaning)",
            rationale="Liquid causes galvanic corrosion and micro-shorts under surface-mount BGAs. Requires immediate professional ultrasonic cleaning and board drying.",
            required_tools=["Torx/Phillips screwdrivers", "99% Isopropyl alcohol", "Ultrasonic cleaner (Lab)"],
            estimated_cost="₹1,500 - ₹5,000 ($20 - $60)",
            estimated_time="1 - 3 business days",
            safety_warnings=[
                "CRITICAL: DO NOT TURN ON THE LAPTOP. DO NOT PLUG IN THE CHARGER.",
                "Leaving power on with liquid present causes instant electrolysis that permanently eats through copper motherboard traces."
            ],
            troubleshooting_steps=[
                "1. IMMEDIATELY disconnect the AC charger.",
                "2. Force power OFF by holding the power button for 10 seconds.",
                "3. Place laptop in an inverted 'tent' (V-shape) on a towel so liquid drains away from the motherboard.",
                "4. DO NOT use a hairdryer (blows liquid deeper under chips and melts plastic keys).",
                "5. DO NOT use rice (starch dust clogs ports and accelerates corrosion).",
                "6. Bring to a board-level repair technician immediately for battery disconnection and ultrasonic alcohol bathing."
            ],
            positive_keywords=["spill", "liquid", "water", "coffee", "tea", "juice", "wet", "poured", "dropped in water"],
            negative_keywords=[],
            common_symptoms=[
                "Liquid was spilled onto keyboard or ports",
                "Laptop turned off immediately after liquid contact",
                "Keys sticking or sizzling sound"
            ],
            co_occurrence_patterns=[
                ["water", "spill"],
                ["coffee", "keyboard"],
                ["liquid", "wet"]
            ]
        ),
        FaultRule(
            fault_id="LAPTOP_PHYSICAL_SCREEN_DAMAGE",
            title="Cracked LCD / OLED Display Panel (Physical Impact)",
            severity="MEDIUM",
            complexity="Medium (Intermediate)",
            recommendation="REPLACE (Display Panel Assembly)",
            rationale="Cracked glass or severed liquid crystal matrix cannot be repaired with software or soldering; it requires replacing the internal display panel.",
            required_tools=["Plastic Spudger / Prying Tool", "Precision Screwdriver PH00", "Replacement OEM Screen Panel"],
            estimated_cost="₹2,500 - ₹6,000 ($30 - $75)",
            estimated_time="30 - 45 minutes",
            safety_warnings=[
                "Disconnect the battery and AC charger before disconnecting the eDP display cable to avoid blowing the motherboard backlight fuse."
            ],
            troubleshooting_steps=[
                "1. Connect laptop to an external monitor via HDMI to verify that GPU and motherboard boot successfully.",
                "2. If external monitor displays the OS, the core system is healthy and only the screen panel requires replacement.",
                "3. Remove bezel frame using a plastic spudger.",
                "4. Unscrew LCD panel mounting brackets and find the panel part number sticker on the back (e.g. NV156FHM-N48).",
                "5. Unclip 30-pin or 40-pin eDP video connector.",
                "6. Install the replacement panel, reconnect eDP cable, test before snapping bezel back."
            ],
            positive_keywords=["screen crack", "crack", "cracked", "broken screen", "shattered", "ink bleed", "black spots", "lines on screen"],
            negative_keywords=[],
            common_symptoms=[
                "Visible crack, hairline fracture, or spiderweb pattern on the display glass",
                "Ink-bleed blotches or black circular voids on the screen",
                "Vertical or horizontal colored lines with fractured glass"
            ],
            co_occurrence_patterns=[
                ["screen", "crack"],
                ["broken", "display"]
            ]
        )
    ],

    # =======================================================================
    # 3. BULB / LIGHTING (LED, CFL, Incandescent)
    # =======================================================================
    "bulb": [
        FaultRule(
            fault_id="BULB_FAILED_LED_DRIVER",
            title="Failed LED Driver Electrolytic Capacitor / Burnt SMD LED Bead",
            severity="LOW",
            complexity="Low (Beginner)",
            recommendation="REPLACE (New Bulb) / REPAIR (DIY Soldering for enthusiasts)",
            rationale="Modern LED bulbs cost ₹70-₹150 ($1-$2). Replacing the bulb is the most economical and reliable action, though shorting a burnt LED diode or replacing the 400V capacitor is a fun 10-minute DIY project.",
            required_tools=["Soldering Iron & Solder", "Small Flathead Screwdriver", "Wire cutter", "Replacement LED Bulb"],
            estimated_cost="₹70 - ₹150 (Replacement) or ₹10 (DIY Solder)",
            estimated_time="5 minutes (Replace) or 20 minutes (DIY Solder)",
            safety_warnings=[
                "Burn hazard: Turn off the light switch and wait 5 minutes for the glass/heatsink to cool.",
                "Never open an LED bulb while it is connected to a live lamp socket."
            ],
            troubleshooting_steps=[
                "1. Turn OFF the light switch.",
                "2. Allow bulb to cool down, then unscrew it from the socket.",
                "3. Test the socket by plugging in a known working bulb to confirm power delivery.",
                "4. If socket is working, the bulb's internal driver or diode board has failed.",
                "5. Optional DIY Repair: Pop off the plastic diffuser globe with a dull knife.",
                "6. Look closely at the yellow LED phosphor squares. If one has a small black dot, that LED burned open.",
                "7. Scrape off the burnt LED and bridge the two solder pads with a tiny blob of solder to complete the series circuit.",
                "8. Snap diffuser back on and test."
            ],
            positive_keywords=["flicker", "flickering", "strobe", "blinking", "not turning on", "dead", "dim", "buzz", "burnt dot", "led", "filament", "burned out"],
            negative_keywords=["switchboard", "breaker"],
            common_symptoms=[
                "Bulb flickers rapidly like a strobe light",
                "Bulb glows extremely dim or only emits a faint orange glimmer",
                "Bulb doesn't turn on at all despite socket having power"
            ],
            co_occurrence_patterns=[
                ["flicker", "dim"],
                ["flicker", "blinking"],
                ["not turning on", "socket"],
                ["buzzing", "bulb"]
            ]
        ),
        FaultRule(
            fault_id="BULB_LOOSE_SOCKET_CONTACT",
            title="Oxidized or Depressed Socket Brass Contact Tang",
            severity="LOW",
            complexity="Low (Beginner)",
            recommendation="REPAIR (DIY)",
            rationale="The brass contact tongue in the center of the holder gets bent downwards over time, failing to touch the bulb bottom contact.",
            required_tools=["Insulated Flathead Screwdriver or Tweezers", "Dry Cloth"],
            estimated_cost="₹0 ($0)",
            estimated_time="5 minutes",
            safety_warnings=[
                "DANGER: TURN OFF THE MAIN ELECTRICAL SWITCH / CIRCUIT BREAKER before touching the inside of a light bulb socket. The center contact is connected directly to 230V/120V Live!"
            ],
            troubleshooting_steps=[
                "1. TURN OFF the wall switch AND the circuit breaker controlling that room.",
                "2. Unscrew the light bulb.",
                "3. Look inside the socket holder. Note the center brass/copper spring tab.",
                "4. If the tab is flattened flush against the bottom, use an insulated screwdriver to gently pry it upwards by 2-3 mm.",
                "5. Wipe any dust or carbon black residue off the contact using a dry cloth.",
                "6. Screw the bulb back in firmly, restore power at the breaker, and turn on the switch."
            ],
            positive_keywords=["loose", "wiggle", "contact", "socket", "works intermittently", "only when twisted", "spark when screwing in", "holder"],
            negative_keywords=["flickering driver"],
            common_symptoms=[
                "Bulb only works when pressed or turned in a specific direction",
                "New bulb also does not work in this specific socket",
                "Sparking or crackling sound inside the socket holder"
            ],
            co_occurrence_patterns=[
                ["socket", "contact"],
                ["wiggle", "light"],
                ["twisted", "holder"]
            ]
        )
    ],

    # =======================================================================
    # 4. SWITCHBOARD / SOCKET / DISTRIBUTION BOX
    # =======================================================================
    "switchboard": [
        FaultRule(
            fault_id="SWITCH_BURNT_CONTACT_ARCING",
            title="Carbonized Switch Rocker Contacts / Terminal Arcing",
            severity="HIGH",
            complexity="Medium (Intermediate)",
            recommendation="REPAIR (DIY Switch Replacement) / SERVICE (Electrician)",
            rationale="Repeated electrical arcing erodes copper switch contacts, creating carbon buildup that increases contact resistance and generates intense heat.",
            required_tools=["Neon Phase Tester / Multimeter", "Insulated Screwdriver (+ & -)", "Replacement Modular Switch (6A/16A)", "Wire Stripper"],
            estimated_cost="₹30 - ₹120 ($0.50 - $1.50) per switch",
            estimated_time="15 - 25 minutes",
            safety_warnings=[
                "HIGH RISK ELECTRICAL HAZARD: Turn off the main MCB (Miniature Circuit Breaker) at the main distribution panel before opening the switchboard plate.",
                "Always verify zero voltage with a calibrated neon tester or digital voltmeter before touching any exposed terminal screws."
            ],
            troubleshooting_steps=[
                "1. Turn OFF the main branch MCB controlling this switchboard.",
                "2. Test a nearby confirmed live socket with your phase tester, then test the target board to ensure power is genuinely DEAD.",
                "3. Carefully pry off the front decorative switch plate cover.",
                "4. Unscrew the two mounting screws holding the modular frame to the wall box.",
                "5. Pull the board gently forward without pulling on wires.",
                "6. Inspect the switch behind: look for black soot, melted plastic casing, or loose terminal screws.",
                "7. Loosen the terminal screws, remove the old switch, and snap the new OEM replacement modular switch into the grid.",
                "8. Strip fresh copper (5mm), insert firmly into terminals, and tighten screws solidly to prevent future loose-connection arcing.",
                "9. Push board back, secure wall screws, and restore MCB power."
            ],
            positive_keywords=["spark", "sparking", "crackling", "buzzing", "burning smell", "black mark", "scorch", "hot switch", "melted", "arcing", "burnt plastic"],
            negative_keywords=[],
            common_symptoms=[
                "Visible blue/yellow spark when flipping the switch ON or OFF",
                "Crackling or sizzling sound behind the switchboard",
                "Switch plate feels warm or hot to the touch",
                "Black soot or scorch marks visible around the switch toggle"
            ],
            co_occurrence_patterns=[
                ["spark", "switch"],
                ["burning", "switchboard"],
                ["crackling", "switch"],
                ["melted", "socket"]
            ]
        ),
        FaultRule(
            fault_id="SOCKET_LOOSE_EARTH_OR_NEUTRAL",
            title="Loose Neutral Terminal / Missing Safety Earth Ground",
            severity="HIGH",
            complexity="Medium (Intermediate)",
            recommendation="REPAIR (DIY) / SERVICE (Licensed Electrician)",
            rationale="Appliances give mild static tingling shocks when safety ground (Earth) is disconnected, or fail to power on when Neutral wire is loose.",
            required_tools=["3-Pin Socket Tester (Receptacle tester)", "Insulated Screwdriver", "Phase tester"],
            estimated_cost="₹0 (Tightening) or ₹80 (New socket)",
            estimated_time="15 minutes",
            safety_warnings=[
                "SHOCK HAZARD: Tingling sensation from metal appliances (refrigerators, PC cabinets, metal kettles) indicates dangerous leakage current without Earth protection.",
                "Turn off main circuit breaker before opening."
            ],
            troubleshooting_steps=[
                "1. Plug a 3-pin socket tester into the faulty socket to immediately diagnose Open Neutral, Open Earth, or Reverse Polarity.",
                "2. Turn OFF main MCB.",
                "3. Open the switchboard faceplate.",
                "4. Inspect the top large pin of the 3-pin socket (Earth terminal, usually Green wire): ensure screw is fully tightened.",
                "5. Inspect the Left terminal (Neutral, usually Black or Blue): ensure the copper wire has not snapped or slipped out.",
                "6. Re-tighten all terminal block screws firmly.",
                "7. Fasten board back to wall, restore power, and re-test."
            ],
            positive_keywords=["shock", "tingle", "tingling", "earth", "neutral", "socket not working", "plug loose", "charger falls out", "spark when plug", "tester shows open earth"],
            negative_keywords=["fan wobble"],
            common_symptoms=[
                "Mild shock or vibration feeling when touching metal laptop casing or appliance body",
                "Socket does not provide power even when switch is toggled ON",
                "Plugs fit loosely and wobble inside the pin sockets"
            ],
            co_occurrence_patterns=[
                ["shock", "socket"],
                ["plug", "loose"],
                ["earth", "tingle"],
                ["socket", "not working"]
            ]
        ),
        FaultRule(
            fault_id="CIRCUIT_BREAKER_TRIP_SHORT",
            title="Short Circuit / Overload Tripping MCB / RCCB",
            severity="CRITICAL",
            complexity="Medium (Intermediate)",
            recommendation="SERVICE (Electrician) / REPAIR (Isolate faulty appliance)",
            rationale="An instant trip occurs when Live directly touches Neutral or Earth, or when high-power heating elements exceed the rated breaker amperage.",
            required_tools=["Multimeter", "Flashlight"],
            estimated_cost="₹0 (Isolation) or ₹250 (New MCB)",
            estimated_time="15 - 30 minutes",
            safety_warnings=[
                "NEVER hold an MCB toggle in the UP position by force or tape it. It trips to protect your house from catastrophic electrical fires."
            ],
            troubleshooting_steps=[
                "1. Identify if the trip is an MCB (Overcurrent/Short Circuit) or an RCCB/ELCB (Earth Leakage).",
                "2. Unplug EVERY device and appliance currently connected to that specific room/circuit.",
                "3. Reset the breaker toggle switch to ON. If it trips immediately with zero appliances plugged in, the fault is inside the wall wiring or switchboard.",
                "4. If the breaker stays ON, plug appliances back in ONE BY ONE until the breaker trips. The last appliance plugged in has an internal dead short!"
            ],
            positive_keywords=["trips", "mcb", "circuit breaker", "tripping", "short circuit", "power cut", "sparks and trips", "rccb", "fuse blown"],
            negative_keywords=[],
            common_symptoms=[
                "Breaker immediately trips with a loud click as soon as device is turned on",
                "Entire room loses power instantly upon plugging in an appliance",
                "Breaker cannot be flipped back up; immediately snaps down"
            ],
            co_occurrence_patterns=[
                ["trip", "mcb"],
                ["short", "circuit"],
                ["breaker", "plugged"],
                ["fuse", "blown"]
            ]
        )
    ]
}


# ---------------------------------------------------------------------------
# DIAGNOSIS ENGINE LOGIC
# ---------------------------------------------------------------------------

class FaultDiagnosisEngine:
    """
    Evaluates detected device and user-described symptoms against deterministic,
    expert-curated rule bases.
    """

    def __init__(self, knowledge_base: Optional[Dict[str, List[FaultRule]]] = None):
        self.kb = knowledge_base or KNOWLEDGE_BASE

    def normalize_device_name(self, raw_device: str) -> str:
        """Map YOLO detections or user labels to canonical device categories."""
        raw = raw_device.strip().lower()
        if any(term in raw for term in ["fan", "ceiling fan", "table fan", "exhaust"]):
            return "fan"
        if any(term in raw for term in ["laptop", "notebook", "computer", "pc"]):
            return "laptop"
        if any(term in raw for term in ["bulb", "light", "lamp", "led", "tube"]):
            return "bulb"
        if any(term in raw for term in ["switchboard", "switch", "socket", "board", "plug", "mcb"]):
            return "switchboard"
        return raw

    def extract_symptom_tokens(self, text: str) -> List[str]:
        """Tokenize and clean symptom description."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [word for word in cleaned.split() if len(word) > 2]

    def score_rule(self, rule: FaultRule, symptom_text: str, symptom_tokens: List[str]) -> Dict[str, Any]:
        """Compute match confidence, evidence, and co-occurrence boosts for a rule."""
        text_lower = symptom_text.lower()
        matched_positive = []
        matched_negative = []
        co_occurrences_found = []

        # 1. Match positive keywords and phrases
        for kw in rule.positive_keywords:
            if " " in kw:
                if kw in text_lower:
                    matched_positive.append(kw)
            else:
                if kw in symptom_tokens or kw in text_lower:
                    matched_positive.append(kw)

        # 2. Check negative (counter-indicator) keywords
        for neg in rule.negative_keywords:
            if neg in text_lower:
                matched_negative.append(neg)

        # 3. High-confidence co-occurrence patterns
        for pattern in rule.co_occurrence_patterns:
            if all(term in text_lower for term in pattern):
                co_occurrences_found.append(" + ".join(pattern))

        # Base score calculation
        if not matched_positive and not co_occurrences_found:
            return {"score": 0.0, "evidence": [], "matches": []}

        # Calculate weighted confidence
        score = min(0.40, len(matched_positive) * 0.15)

        # Big boost for co-occurring symptom pairs
        if co_occurrences_found:
            score += len(co_occurrences_found) * 0.35

        # Penalty for negative matches
        if matched_negative:
            score -= len(matched_negative) * 0.25

        score = max(0.0, min(0.99, score))

        evidence = []
        if matched_positive:
            evidence.append(f"Reported symptoms directly match indicator terms: {', '.join(set(matched_positive))}")
        if co_occurrences_found:
            evidence.append(f"Strong symptom co-occurrence pattern identified: [{', '.join(co_occurrences_found)}]")
        if matched_negative:
            evidence.append(f"Contains counter-indicator terms: {', '.join(set(matched_negative))}")

        return {
            "score": round(score, 2),
            "evidence": evidence,
            "matches": list(set(matched_positive))
        }

    def diagnose(
        self,
        device: str,
        user_symptoms: str,
        detection_confidence: Optional[float] = None,
        detection_bbox: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Produce a complete structured diagnostic report combining object detection and symptom rules.
        """
        canonical_device = self.normalize_device_name(device)
        rules = self.kb.get(canonical_device, [])

        if not rules:
            return {
                "status": "error",
                "device": device,
                "message": f"Device category '{device}' is not yet supported in the diagnostic knowledge base. Supported: {list(self.kb.keys())}"
            }

        symptom_tokens = self.extract_symptom_tokens(user_symptoms)

        scored_candidates = []
        for rule in rules:
            analysis = self.score_rule(rule, user_symptoms, symptom_tokens)
            if analysis["score"] > 0:
                scored_candidates.append({
                    "rule": rule,
                    "score": analysis["score"],
                    "evidence": analysis["evidence"],
                    "matches": analysis["matches"]
                })

        # Sort candidate faults by score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        if not scored_candidates:
            # Fallback if no specific keyword matched: suggest baseline diagnostic checks
            primary_rule = rules[0]
            confidence_level = 0.35
            diagnosis_note = "Generic diagnostic check: Symptoms provided did not match a specific single failure mode with high confidence."
            evidence = ["User-described symptoms were broad or nonspecific. Suggesting primary systematic triage steps."]
        else:
            top = scored_candidates[0]
            primary_rule = top["rule"]
            confidence_level = top["score"]
            diagnosis_note = f"High confidence rule match for '{primary_rule.title}' based on provided symptoms."
            evidence = top["evidence"]

        # Build primary diagnosis block
        primary_diag = {
            "fault_id": primary_rule.fault_id,
            "fault_title": primary_rule.title,
            "confidence_score": confidence_level,
            "severity": primary_rule.severity,
            "repair_complexity": primary_rule.complexity,
            "recommendation": primary_rule.recommendation,
            "recommendation_rationale": primary_rule.rationale,
            "estimated_diy_time": primary_rule.estimated_time,
            "estimated_parts_cost": primary_rule.estimated_cost,
            "required_tools": primary_rule.required_tools,
            "safety_warnings": primary_rule.safety_warnings,
            "troubleshooting_steps": primary_rule.troubleshooting_steps,
            "supporting_evidence": evidence
        }

        # Build alternative secondary diagnoses
        alternatives = []
        for item in scored_candidates[1:3]:
            alt_rule = item["rule"]
            alternatives.append({
                "fault_id": alt_rule.fault_id,
                "fault_title": alt_rule.title,
                "confidence_score": item["score"],
                "recommendation": alt_rule.recommendation,
                "supporting_evidence": item["evidence"]
            })

        result = {
            "status": "success",
            "device": canonical_device,
            "original_detection": {
                "detected_device": device,
                "detection_confidence": detection_confidence,
                "bounding_box": detection_bbox
            },
            "user_symptoms": user_symptoms,
            "diagnosis_note": diagnosis_note,
            "primary_diagnosis": primary_diag,
            "alternative_diagnoses": alternatives,
            "critical_safety_alert": primary_rule.safety_warnings[0] if primary_rule.safety_warnings else "Always disconnect power before inspection."
        }

        return result

    def diagnose_questionnaire(self, questionnaire_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a questionnaire output payload (with device_type, detected_findings, answers)
        and populates:
          - 'safety_alert': bool (set to True if hazards, liquid, or high-risk faults exist)
          - 'requires_professional_attention': bool
          - 'diagnosis': detailed structured diagnosis object
          - 'repair_recommendation': actionable recommendation object with cost & complexity
        """
        result = dict(questionnaire_payload)
        device = result.get("device_type", "laptop")
        findings = result.get("detected_findings", [])
        answers = result.get("answers", {})
        det_conf = result.get("detection_confidence", 0.90)

        # 1. Synthesize symptom string from visual findings and questionnaire answers
        symptom_parts = []
        if findings:
            symptom_parts.append(f"Visual detection findings: {', '.join(findings)}")

        ignored_answers = {"", "no", "none", "not tested", "normal", "unknown", "n/a", "none of the above"}
        for question_key, ans in answers.items():
            if str(ans).strip().lower() not in ignored_answers:
                clean_key = question_key.replace("_", " ")
                symptom_parts.append(f"{clean_key}: {ans}")

        combined_symptoms = ". ".join(symptom_parts)
        if not combined_symptoms:
            combined_symptoms = "Device is not operating as expected."

        # 2. Safety & Hazard Analysis
        hazards = str(answers.get("hazards", "")).strip().lower()
        possible_cause = str(answers.get("possible_cause", "")).strip().lower()

        safety_alert = False
        requires_pro = False
        safety_reasons = []

        if hazards not in ["no", "", "none", "n/a"]:
            safety_alert = True
            requires_pro = True
            safety_reasons.append(f"Reported active hazard: {answers.get('hazards')}")

        if "liquid" in possible_cause or "water" in possible_cause:
            safety_alert = True
            requires_pro = True
            safety_reasons.append("Liquid exposure causes severe corrosion and micro-shorts under surface-mount ICs. Requires ultrasonic board cleaning.")

        if any(term in combined_symptoms.lower() for term in ["spark", "smoke", "shock", "burning", "burnt"]):
            safety_alert = True
            requires_pro = True
            safety_reasons.append("Thermal/electrical hazard detected (sparking, smoke, or burning smell).")

        # 3. Execute core diagnosis
        base_diag = self.diagnose(
            device=device,
            user_symptoms=combined_symptoms,
            detection_confidence=det_conf
        )

        primary = base_diag["primary_diagnosis"]
        alternatives = base_diag.get("alternative_diagnoses", [])

        # If severity is CRITICAL or DANGEROUS, elevate flags
        if primary["severity"] in ["CRITICAL", "HIGH"] or "dangerous" in primary["repair_complexity"].lower():
            safety_alert = True
            if "service" in primary["recommendation"].lower() or "dangerous" in primary["repair_complexity"].lower():
                requires_pro = True

        # Check visual findings (e.g. screen crack) to augment secondary faults or recommendation
        if any("crack" in f.lower() for f in findings) and primary["fault_id"] != "LAPTOP_PHYSICAL_SCREEN_DAMAGE":
            alternatives.insert(0, {
                "fault_id": "LAPTOP_PHYSICAL_SCREEN_DAMAGE",
                "fault_title": "Cracked LCD / OLED Display Panel (Physical Impact)",
                "confidence_score": 0.95,
                "recommendation": "REPLACE (Display Panel Assembly)",
                "supporting_evidence": ["Visual object detection identified physical screen fracture."]
            })

        # 4. Format populated diagnosis block
        result["safety_alert"] = safety_alert
        result["requires_professional_attention"] = requires_pro

        result["diagnosis"] = {
            "primary_fault": primary["fault_title"],
            "fault_id": primary["fault_id"],
            "confidence": primary["confidence_score"],
            "severity": primary["severity"],
            "supporting_evidence": primary["supporting_evidence"] + safety_reasons,
            "safety_instructions": primary["safety_warnings"],
            "troubleshooting_steps": primary["troubleshooting_steps"],
            "secondary_faults": [
                {
                    "fault_id": alt["fault_id"],
                    "fault_title": alt["fault_title"],
                    "confidence": alt["confidence_score"],
                    "recommendation": alt["recommendation"]
                }
                for alt in alternatives
            ]
        }

        # 5. Format populated repair recommendation block
        result["repair_recommendation"] = {
            "action": primary["recommendation"],
            "complexity": primary["repair_complexity"],
            "rationale": primary["recommendation_rationale"],
            "estimated_cost": primary["estimated_parts_cost"],
            "estimated_time": primary["estimated_diy_time"],
            "required_tools": primary["required_tools"]
        }

        return result

    def get_supported_devices(self) -> List[str]:
        return list(self.kb.keys())

    def get_device_symptom_checklist(self, device: str) -> List[str]:
        canonical = self.normalize_device_name(device)
        rules = self.kb.get(canonical, [])
        checklist = []
        for r in rules:
            checklist.extend(r.common_symptoms)
        return checklist


# Global engine singleton
engine = FaultDiagnosisEngine()


def run_diagnosis(device: str, symptoms: str) -> Dict[str, Any]:
    """Simple API helper function."""
    return engine.diagnose(device=device, user_symptoms=symptoms)


def process_questionnaire(questionnaire_data: Dict[str, Any]) -> Dict[str, Any]:
    """Integrates directly with questionnaire output dictionaries."""
    return engine.diagnose_questionnaire(questionnaire_data)


