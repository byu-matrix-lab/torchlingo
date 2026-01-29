document.addEventListener('DOMContentLoaded', function () {
  // Wait until mermaid is loaded, then initialize.
  function initMermaid() {
    if (typeof mermaid !== 'undefined') {
      try {
        mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });
      } catch (e) {
        console.warn('Mermaid initialization failed:', e);
      }
    } else {
      // Retry a few times if the script hasn't loaded yet
      retries -= 1;
      if (retries > 0) {
        setTimeout(initMermaid, 150);
      } else {
        console.warn('Mermaid library not available after retries.');
      }
    }
  }

  var retries = 5;
  initMermaid();
});

// Optional: Sanitize Jupyter-rendered Mermaid error details which can pollute
// the page with large parse stacks. We replace verbose error blocks with a
// short, friendly message so pages stay readable for students.
(function sanitizeMermaidErrors() {
  function clean() {
    const els = document.querySelectorAll('.jp-RenderedMermaid.jp-mod-warning');
    els.forEach((el) => {
      // Replace detailed error content with a small, non-intrusive message
      if (!el.dataset.cleaned) {
        const msg = document.createElement('div');
        msg.className = 'mermaid-error-brief';
        msg.textContent = '⚠️ Mermaid diagram could not be rendered. Check diagram syntax.';
        // keep figure/img if present, otherwise just show message
        const fig = el.querySelector('figure');
        el.innerHTML = '';
        if (fig) el.appendChild(fig);
        el.appendChild(msg);
        el.dataset.cleaned = '1';
      }
    });
  }

  // Run a few times to catch dynamic notebook-rendered content
  for (let i = 0; i < 6; i++) setTimeout(clean, 200 * i);
  // Also observe mutations for late-added errors
  const mo = new MutationObserver(clean);
  mo.observe(document.body, { childList: true, subtree: true });
})();
