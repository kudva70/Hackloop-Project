"""
AI Repair Assistant - Legacy Rules Archive
Archived rules for fan, bulb, switchboard, and mouse.
NOT imported or used by fault_engine.py (MVP active scope: Laptops & PCs only).
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class LegacyFaultRule:
    fault_id: str
    title: str
    severity: str
    complexity: str
    recommendation: str
    rationale: str
    required_tools: List[str]
    estimated_cost: str
    estimated_time: str
    safety_warnings: List[str]
    troubleshooting_steps: List[str]
    positive_keywords: List[str]
    negative_keywords: List[str]
    common_symptoms: List[str]
    co_occurrence_patterns: List[List[str]]


LEGACY_KNOWLEDGE_BASE = {
    "fan": [
        LegacyFaultRule(
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
        LegacyFaultRule(
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
        LegacyFaultRule(
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
        LegacyFaultRule(
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
    # 3. BULB / LIGHTING (LED, CFL, Incandescent)
    # =======================================================================
    "bulb": [
        LegacyFaultRule(
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
        LegacyFaultRule(
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
        LegacyFaultRule(
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
        LegacyFaultRule(
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
        LegacyFaultRule(
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
    ],

    # =======================================================================
    # 5. MOUSE / POINTING DEVICE (Wired / Wireless / Bluetooth)
    # =======================================================================
    "mouse": [
        LegacyFaultRule(
            fault_id="MOUSE_DOUBLE_CLICK_MICROSWITCH",
            title="Worn-Out Left/Right Button Microswitch (Double-Click on Single Click)",
            severity="MEDIUM",
            complexity="Medium (Intermediate)",
            recommendation="REPAIR (DIY Solder) / REPLACE (New Mouse)",
            rationale="The tactile microswitch inside the mouse button loses its spring-return force after ~1-3 million actuations, causing a ghost secondary click on every physical press.",
            required_tools=["Soldering Iron", "Desoldering Pump or Braid", "Replacement OMRON D2FC-F-7N Microswitch", "Small Phillips Screwdriver"],
            estimated_cost="\u20b930 - \u20b9120 (Microswitch) or \u20b9400 (New basic mouse)",
            estimated_time="20 - 40 minutes",
            safety_warnings=[
                "Unplug the mouse (or remove batteries for wireless) before opening.",
                "Identify the exact PCB microswitch model number before ordering: D2FC-F-7N (10M) for Omron, or TTC Gold for smoother feel."
            ],
            troubleshooting_steps=[
                "1. Verify the fault: Open any text editor and single-click; if text gets selected (double-click highlight), the microswitch is the culprit.",
                "2. Go to Mouse Settings -> Test mouse buttons to confirm the ghosting.",
                "3. Remove the mouse feet (adhesive pads on the bottom) to find hidden screws.",
                "4. Pry open the mouse shell gently with a spudger.",
                "5. Locate the two front microswitches on the PCB.",
                "6. Heat the two solder pads of the faulty switch from the PCB underside and remove the old switch.",
                "7. Solder in the new OMRON D2FC-F-7N microswitch and reassemble.",
                "8. Test clicking in the OS mouse button diagnostics tool."
            ],
            positive_keywords=["double click", "double clicking", "double-click", "single click selects", "drag and drop fails", "misclick", "unintended double", "clicking twice", "ghost click", "self clicking"],
            negative_keywords=["scroll", "sensor", "cursor"],
            common_symptoms=[
                "Single physical click registers as two rapid clicks",
                "Drag-and-drop randomly releases mid-drag",
                "Files/folders open on single click instead of requiring a double-click"
            ],
            co_occurrence_patterns=[
                ["double", "click"],
                ["single", "select"],
                ["drag", "drop"],
                ["ghost", "click"]
            ]
        ),
        LegacyFaultRule(
            fault_id="MOUSE_SENSOR_STUTTER",
            title="Optical/Laser Sensor Contamination or Surface Incompatibility",
            severity="LOW",
            complexity="Low (Beginner)",
            recommendation="REPAIR (DIY Clean)",
            rationale="Dust, lint, or oil accumulation on the sensor lens causes erratic cursor jumps. A simple clean restores perfect tracking.",
            required_tools=["Compressed Air Can", "Cotton Swab", "99% Isopropyl Alcohol", "Lint-Free Cloth"],
            estimated_cost="\u20b90 ($0)",
            estimated_time="5 minutes",
            safety_warnings=[
                "Do not use excessive liquid on the sensor; one lightly-dampened cotton swab is sufficient.",
                "Use an appropriate mouse pad (solid non-reflective surface). Glass and mirror surfaces confuse laser sensors."
            ],
            troubleshooting_steps=[
                "1. Lift the mouse and observe the sensor LED glow. If it flickers or is dim, dust is blocking the lens.",
                "2. Blow one short burst of compressed air directly into the sensor opening.",
                "3. Lightly dab a cotton swab in isopropyl alcohol and gently wipe the sensor lens in a circular motion.",
                "4. Allow 30 seconds for alcohol to evaporate.",
                "5. Test on a high-contrast mouse pad (light gray or cloth recommended).",
                "6. If cursor still stutters on any surface, the sensor ASIC may be internally damaged."
            ],
            positive_keywords=["cursor jumping", "jumping cursor", "stuttering", "stutter", "freezing cursor", "cursor freeze", "sensor", "erratic movement", "random cursor", "drift", "glitching cursor", "tracking issue"],
            negative_keywords=["click", "scroll", "battery"],
            common_symptoms=[
                "Cursor jumps or teleports randomly while moving the mouse",
                "Cursor freezes momentarily then catches up",
                "Mouse works fine on cloth pad but stutters on hard desk surface"
            ],
            co_occurrence_patterns=[
                ["cursor", "jump"],
                ["stutter", "move"],
                ["freeze", "cursor"],
                ["erratic", "movement"]
            ]
        ),
        LegacyFaultRule(
            fault_id="MOUSE_SCROLL_WHEEL_SKIP",
            title="Dirty or Worn Scroll Wheel Encoder",
            severity="LOW",
            complexity="Low (Beginner)",
            recommendation="REPAIR (DIY Clean or Encoder Swap)",
            rationale="Rotary encoder inside the scroll wheel accumulates dust and graphite causing it to skip steps or scroll in reverse. A cleaning or \u20b920 encoder swap fixes it.",
            required_tools=["Compressed Air Can", "Cotton Swab", "99% Isopropyl Alcohol", "Screwdriver (for encoder swap)"],
            estimated_cost="\u20b90 (clean) or \u20b920 - \u20b960 (new encoder)",
            estimated_time="10 - 20 minutes",
            safety_warnings=[
                "Disconnect the mouse before opening the shell.",
                "Do not spin the scroll wheel at high speed with the encoder exposed; the encoder teeth are delicate."
            ],
            troubleshooting_steps=[
                "1. Confirm the fault: open a long webpage and scroll slowly. If it jumps multiple lines per notch or scrolls backward, the encoder is failing.",
                "2. Spray a short burst of compressed air around the scroll wheel gap from both sides.",
                "3. If available, apply a tiny drop of electronics contact cleaner (Deoxit D5) around the encoder shaft.",
                "4. Scroll the wheel rapidly 20-30 times to distribute the cleaner inside the encoder.",
                "5. Wait 5 minutes and test. If skipping continues, open the mouse shell.",
                "6. Desolder the rotary encoder (3 or 5 pin) and replace with an identical part."
            ],
            positive_keywords=["scroll", "scrolling", "scroll skips", "scroll jumps", "scroll backwards", "reverse scroll", "scroll not smooth", "wheel skip", "wheel jump", "encoder"],
            negative_keywords=["click", "cursor", "sensor"],
            common_symptoms=[
                "Scroll wheel skips or jumps multiple lines per single notch click",
                "Page scrolls in the opposite direction unexpectedly",
                "Scroll wheel feels gritty, resistant, or produces scratchy sound while rolling"
            ],
            co_occurrence_patterns=[
                ["scroll", "skip"],
                ["scroll", "backwards"],
                ["wheel", "jump"],
                ["scroll", "rough"]
            ]
        )
    ]
}


# This project currently diagnoses laptop/PC hardware only. The older fan, bulb,
# switchboard and unrelated device rules above are deliberately excluded from the
# active knowledge base so they cannot be selected by the questionnaire or engine.

