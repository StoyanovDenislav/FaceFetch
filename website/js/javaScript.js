const Type_Class_List = ["alert", "granted", "warning"];
class Activity_Log extends HTMLElement {
    connectedCallback() {
        const type = this.getAttribute("type");
        const type_class = this.getAttribute("type_class");
        const desc = this.getAttribute("desc");
        const score = this.getAttribute("score");
        const datetime = this.getAttribute("datetime");

        this.innerHTML = `
      <div class="activity_log_cont">
        <div class="activity_log_type ${type_class}">
          <span>${type}</span>
        </div>

        <div class="activity_log_desc">
          <span>${desc}</span>
        </div>

        <div class="activity_log_score">
          <span>Confidence:</span>
          <span>${score}%</span>
        </div>

        <div class="activity_log_datetime">
          <span>Logged at:</span>
          <span>${datetime}</span>
        </div>
      </div>
    `;
    }
}

customElements.define("activity-log", Activity_Log);


const data = [
    {
        type: "Intruder Detected",
        type_class: 0,
        desc: "Facefetch system detected an possible intruder! Please take action immediately!",
        score: 92,
        datetime: "2026-01-15 14:32"
    },
    {
        type: "Known Face Detected",
        type_class: 1,
        desc: "Familiar face has been recognized. All is okay.",
        score: 98,
        datetime: "2026-01-15 14:40"
    }
];

function LoadData(containerId)
{
    let container = document.getElementById(containerId);

    data.forEach(item => {
        const log = document.createElement("activity-log");

        log.setAttribute("type", item.type);
        log.setAttribute("type_class", Type_Class_List[item.type_class]);
        log.setAttribute("desc", item.desc);
        log.setAttribute("score", item.score);
        log.setAttribute("datetime", item.datetime);

        container.appendChild(log);
    });
}

LoadData('dashboard_log_container');