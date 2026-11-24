'use strict';
const scenarios=[
  {
    "id": "dynamics",
    "tab": "Dynamics change",
    "tag": "Learning a new drag value",
    "title": "Adapting when drag changes.",
    "cohort": "30 starts per direction",
    "description": "At 20 seconds, drag changes and all four landmarks move. Watch CoPH estimate the new drag and compare its motion with a controller that keeps the old value.",
    "steps": [
      "The drag changes from 0.25 to 0.45.",
      "Measured motion no longer matches the old model.",
      "CoPH estimates drag and updates its control.",
      "Compare progress and return after the change."
    ],
    "limit": "The demo uses seed 87000. The study also tests drag decreasing to 0.10. Velocities and an environmental field were not saved for this run.",
    "clock": "Each frame shows mission time. Playback is compressed, with more frames near the change.",
    "events": [
      [
        "Before change",
        19.9
      ],
      [
        "Dynamics change",
        20
      ],
      [
        "Identification updates",
        22.4
      ],
      [
        "End of run",
        40
      ]
    ]
  },
  {
    "id": "hazards",
    "tab": "Different preferences",
    "tag": "Same terrain, different risk weights",
    "title": "The scout crosses; the carrier takes a detour.",
    "cohort": "Four hazard scenes",
    "description": "The carrier starts 25 seconds after the scout and puts more weight on risk. The scout sends hazard estimates, which the carrier uses to choose its route. Dashed paths show what happens when it receives only the scout’s advice.",
    "steps": [
      "The carrier has not seen the hazardous corridor.",
      "The scout sends estimates from its measurements.",
      "The carrier evaluates the route with its own risk weight.",
      "Watch where the carrier goes and how much risk it encounters."
    ],
    "limit": "The demo uses seed 79000. Raw readings match the field’s navigation results in this pilot. Receiver fitting has not shown an extra benefit here.",
    "clock": "Both agents use the same clock; the carrier starts at 25 s. Surfaces show the model used at that time. Regions without enough local measurements are masked.",
    "events": [
      [
        "Scout starts",
        0
      ],
      [
        "Carrier starts",
        25
      ],
      [
        "Shared observations",
        30
      ],
      [
        "Later route",
        55
      ]
    ]
  },
  {
    "id": "reciprocity",
    "tab": "Reciprocal discovery",
    "tag": "Warnings in both directions",
    "title": "A changed route leads to a second warning.",
    "cohort": "One controlled crossing",
    "description": "Two agents start at opposite ends of a network. They know the corridors but not which ones are blocked. One warning redirects a teammate, which reaches a second blockage sooner and sends a warning back.",
    "steps": [
      "The corridor network is known; the closures are not.",
      "One agent discovers a blockage and sends a warning.",
      "Its teammate changes route and finds another blockage.",
      "The return warning saves further travel."
    ],
    "limit": "Seed 76000. Solid paths show two-way communication; dashed paths show no communication. The benefit depends on the layout and warning times. Inter-agent collisions are not modeled in this demo.",
    "clock": "Frames show mission time and model versions. Surface height is shaping energy, not terrain height. The full map is shown for the viewer, not given to the agents.",
    "events": [
      [
        "Initial routes",
        0
      ],
      [
        "First warning",
        4.8
      ],
      [
        "Return warning",
        21.12
      ],
      [
        "Later progress",
        70
      ]
    ]
  },
  {
    "id": "encounters",
    "tab": "Symmetric encounter",
    "tag": "Taking turns through a crossing",
    "title": "Breaking a four-agent deadlock.",
    "cohort": "30 starts per encounter type",
    "description": "Four agents try to swap places through the center. With the old protection layer, each can wait for the others and nobody passes. The final layer adds a fixed priority rule so they can take turns.",
    "steps": [
      "All four agents approach the same crossing.",
      "Collision constraints can make everyone wait.",
      "A fixed priority rule makes some agents yield.",
      "Compare completion and the distances between agents."
    ],
    "limit": "Seed 300. Yielding is programmed, not learned. Forces, yielding events and model updates were not saved for this run.",
    "clock": "Positions are shown at 0, 0.1, 0.5, …, 39.7 s. These samples do not show the minimum separation between frames.",
    "events": [
      [
        "Initial state",
        0
      ],
      [
        "Approaching crossing",
        4.1
      ],
      [
        "Passing through",
        14.1
      ],
      [
        "End of run",
        39.7
      ]
    ]
  },
  {
    "id": "uncertainty",
    "tab": "Sparse evidence",
    "tag": "Choosing a path from a few measurements",
    "title": "Which path is safer between the measurements?",
    "cohort": "12 groups × 400 path pairs",
    "description": "A few measurements leave gaps between sampled locations. The field, raw-reading and costmap methods estimate the cost of the same two paths and choose one. Watch their choices, uncertainty labels and the extra cost of a wrong choice.",
    "steps": [
      "Risk is measured at only a few locations.",
      "Each method estimates risk along two candidate paths.",
      "It chooses a path and checks whether that choice is consistent across uncertainty samples.",
      "Compare the choice with the cheaper path under the true field."
    ],
    "limit": "This is a sequence of saved-state path choices, not a navigation run. The video shows the first 12 queries. The original uncertainty samples were not saved.",
    "clock": "Frames are numbered by query, not mission time. Surface height shows estimated risk. “Certified” means 95% of uncertainty samples agree on a choice; it is not a safety guarantee.",
    "events": [
      [
        "First query",
        0
      ],
      [
        "Fourth query",
        3
      ],
      [
        "Eighth query",
        7
      ],
      [
        "Last displayed query",
        11
      ]
    ]
  }
];
const video=document.getElementById('scenario-video');const speed=document.getElementById('playback-speed');speed.addEventListener('change',()=>{video.playbackRate=Number(speed.value)});let current=0;let provenance;
const tabs=document.getElementById('scenario-tabs');
scenarios.forEach((s,i)=>{const b=document.createElement('button');b.textContent=s.tab;b.id=`tab-${s.id}`;b.type='button';b.setAttribute('role','tab');b.setAttribute('aria-controls','scenario-video');b.addEventListener('click',()=>select(i));tabs.append(b)});
function updateEvents(){const ready=Boolean(provenance&&video.readyState>=1&&video.currentSrc.endsWith(`/assets/${scenarios[current].id}.mp4`));[...document.querySelectorAll('#event-buttons button')].forEach(b=>b.disabled=!ready)}
video.addEventListener('loadedmetadata',updateEvents);
function seek(s,value){if(!provenance||video.readyState<1)return;const frames=provenance.frames[s.id];if(!frames?.length)return;const key=s.id==='uncertainty'?'query_index':'mission_seconds';const nearest=frames.reduce((a,b)=>Math.abs(b[key]-value)<Math.abs(a[key]-value)?b:a);video.currentTime=nearest.video_seconds;video.pause()}
function select(i){current=i;const s=scenarios[i];[...tabs.children].forEach((b,k)=>{b.setAttribute('aria-selected',String(i===k));b.tabIndex=i===k?0:-1});video.pause();video.src=`assets/${s.id}.mp4`;video.poster=`assets/${s.id}.png`;video.setAttribute('aria-label',s.title);video.load();video.playbackRate=Number(speed.value);document.getElementById('scenario-tag').textContent=s.tag;document.getElementById('scenario-title').textContent=s.title;document.getElementById('scenario-description').textContent=s.description;document.getElementById('scenario-cohort').textContent=s.cohort;document.getElementById('scenario-limit').textContent=s.limit;document.getElementById('scenario-clock').textContent=s.clock;document.getElementById('gif-link').href=`assets/${s.id}.gif`;document.getElementById('frames-link').href=`records.html#${s.id}`;document.getElementById('summary-link').href=`results.html?v=20261124-copy#${s.id}`;
 const box=document.getElementById('mechanism');box.replaceChildren();s.steps.forEach((t,k)=>{const d=document.createElement('div'),h=document.createElement('h4'),p=document.createElement('p');h.textContent=['Problem','Observation','Response','What to watch'][k];p.textContent=t;d.append(h,p);box.append(d)});
 const events=document.getElementById('event-buttons');events.replaceChildren();s.events.forEach(([label,t])=>{const b=document.createElement('button');b.type='button';b.textContent=label;b.disabled=true;b.addEventListener('click',()=>seek(s,t));events.append(b)});history.replaceState(null,'',`#${s.id}`)}
tabs.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();const focused=[...tabs.children].indexOf(document.activeElement);select(((focused>=0?focused:current)+(e.key==='ArrowRight'?1:-1)+scenarios.length)%scenarios.length);tabs.children[current].focus()}});
fetch('assets/provenance.json').then(r=>{if(!r.ok)throw Error('Provenance unavailable');return r.json()}).then(p=>{provenance=p;updateEvents()}).catch(()=>{document.getElementById('scenario-clock').textContent+=' Frame provenance is temporarily unavailable; the video controls remain usable.'});
const initial=scenarios.findIndex(s=>`#${s.id}`===location.hash);select(initial<0?0:initial);
