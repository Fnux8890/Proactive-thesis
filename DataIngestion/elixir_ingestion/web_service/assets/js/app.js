// Import dependencies
import "phoenix_html"
// Import local files
//
// Local files can be imported directly using relative paths, for example:
// import "./user_socket.js"
import {Socket} from "phoenix"
import {LiveSocket} from "phoenix_live_view"

let csrfToken = document.querySelector("meta[name='csrf-token']").getAttribute("content")

// Define any hooks for LiveView
let Hooks = {}

// Initialize LiveSocket
let liveSocket = new LiveSocket("/live", Socket, {
  params: {_csrf_token: csrfToken},
  hooks: Hooks
})

// Show progress bar on live navigation and form submits
let progressBar = document.getElementById("progress-bar")

window.addEventListener("phx:page-loading-start", _info => {
  if (progressBar) progressBar.style.width = "45%"
})

window.addEventListener("phx:page-loading-stop", _info => {
  if (progressBar) {
    progressBar.style.width = "100%"
    setTimeout(() => {
      progressBar.style.width = "0%"
    }, 200)
  }
})

// Sidebar functionality
document.addEventListener("DOMContentLoaded", function() {
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebar = document.getElementById("sidebar");
  const mainContent = document.getElementById("main-content");
  let sidebarOpen = false;

  if (sidebarToggle && sidebar && mainContent) {
    // Toggle sidebar when button is clicked
    sidebarToggle.addEventListener("click", function() {
      sidebarOpen = !sidebarOpen;
      if (sidebarOpen) {
        sidebar.classList.add("open");
        mainContent.classList.add("sidebar-open");
        sidebarToggle.innerHTML = '<i class="fas fa-times text-xl"></i>';
      } else {
        sidebar.classList.remove("open");
        mainContent.classList.remove("sidebar-open");
        sidebarToggle.innerHTML = '<i class="fas fa-bars text-xl"></i>';
      }
    });

    // Close sidebar when clicking anywhere in main content (mobile-friendly behavior)
    mainContent.addEventListener("click", function(e) {
      if (sidebarOpen && window.innerWidth < 768) {
        sidebarOpen = false;
        sidebar.classList.remove("open");
        mainContent.classList.remove("sidebar-open");
        sidebarToggle.innerHTML = '<i class="fas fa-bars text-xl"></i>';
      }
    });
  }

  // Add pipeline stage hover effects
  const pipelineStages = document.querySelectorAll('.pipeline-stage');
  pipelineStages.forEach(stage => {
    stage.addEventListener('mouseenter', () => {
      stage.classList.add('bg-blue-50');
    });
    stage.addEventListener('mouseleave', () => {
      stage.classList.remove('bg-blue-50');
    });
  });
});

// connect if there are any LiveViews on the page
liveSocket.connect()

// expose liveSocket on window for web console debug logs and latency simulation:
// >> liveSocket.enableDebug()
// >> liveSocket.enableLatencySim(1000)  // enabled for duration of browser session
// >> liveSocket.disableLatencySim()
window.liveSocket = liveSocket
