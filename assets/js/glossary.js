class GlossaryManager {
  constructor() {
    this.glossary = {};
    this.tooltip = null;
    this.hoverTimeout = null;
    this.hideTimeout = null;
    this.processedTerms = new Set(); // Track terms that have been processed once
  }

  async init() {
    try {
      const response = await fetch('/assets/data/glossary.json');
      this.glossary = await response.json();
      this.processPage();
      this.setupTooltip();
    } catch (error) {
      console.warn('Could not load glossary:', error);
    }
  }

  processPage() {
    const contentArea = document.querySelector('.post-content, .page-content, main');
    if (!contentArea) return;

    if (contentArea.closest('.no-glossary')) return;

    this.processTextNodes(contentArea);
  }

  processTextNodes(element) {
    const skipTags = ['SCRIPT', 'STYLE', 'CODE', 'PRE', 'A', 'BUTTON', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'SUMMARY'];
    if (skipTags.includes(element.tagName)) return;
    if (element.classList && element.classList.contains('no-glossary')) return;
    if (element.classList && (
      element.classList.contains('post-title') || 
      element.classList.contains('site-title') ||
      element.classList.contains('page-title')
    )) return;

    if (element.nodeType === Node.TEXT_NODE) {
      this.processTextNode(element);
    } else {
      Array.from(element.childNodes).forEach(child => {
        this.processTextNodes(child);
      });
    }
  }

  processTextNode(textNode) {
    let text = textNode.textContent;
    let hasMatches = false;

    const termMap = {};
    Object.entries(this.glossary).forEach(([entryKey, entry]) => {
      termMap[entry.term] = {
        definition: entry.definition,
        firstOnly: entry.first_only || false,
        entryKey: entryKey // Track which glossary entry this belongs to
      };
      if (entry.variations) {
        entry.variations.forEach(variation => {
          termMap[variation] = {
            definition: entry.definition,
            firstOnly: entry.first_only || false,
            entryKey: entryKey // Same entry key for variations
          };
        });
      }
    });

    // Sort terms by length (longest first) to handle overlapping matches better
    const sortedTerms = Object.keys(termMap).sort((a, b) => b.length - a.length);

    // Create document fragment to build new content
    const fragment = document.createDocumentFragment();
    let lastIndex = 0;

    const matches = [];
    sortedTerms.forEach(term => {
      const termInfo = termMap[term];
      
      if (termInfo.firstOnly && this.processedTerms.has(termInfo.entryKey)) {
        return;
      }

      const regex = new RegExp(`\\b${this.escapeRegex(term)}\\b`, 'g');
      let match;
      let foundFirstMatch = false;
      
      while ((match = regex.exec(text)) !== null) {
        if (termInfo.firstOnly && foundFirstMatch) {
          break;
        }
        
        matches.push({
          start: match.index,
          end: match.index + match[0].length,
          text: match[0],
          definition: termInfo.definition,
          firstOnly: termInfo.firstOnly,
          entryKey: termInfo.entryKey
        });
        
        if (termInfo.firstOnly) {
          foundFirstMatch = true;
        }
      }
    });

    matches.sort((a, b) => a.start - b.start);
    const validMatches = [];
    let lastEnd = 0;
    matches.forEach(match => {
      if (match.start >= lastEnd) {
        validMatches.push(match);
        lastEnd = match.end;
      }
    });

    validMatches.forEach(match => {
      if (match.start > lastIndex) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.start)));
      }

      const span = document.createElement('span');
      span.className = 'glossary-term';
      span.setAttribute('data-definition', match.definition);
      span.textContent = match.text;
      fragment.appendChild(span);

      if (match.firstOnly) {
        this.processedTerms.add(match.entryKey);
      }

      lastIndex = match.end;
      hasMatches = true;
    });

    if (lastIndex < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
    }

    if (hasMatches) {
      textNode.parentNode.replaceChild(fragment, textNode);
    }
  }

  setupTooltip() {
    this.tooltip = document.createElement('div');
    this.tooltip.className = 'glossary-tooltip';
    document.body.appendChild(this.tooltip);

    document.addEventListener('mouseover', (e) => {
      if (e.target.classList.contains('glossary-term') || e.target.classList.contains('def')) {
        this.showTooltip(e.target, e);
      }
    });

    document.addEventListener('mouseout', (e) => {
      if (e.target.classList.contains('glossary-term') || e.target.classList.contains('def')) {
        this.hideTooltip();
      }
    });

    document.addEventListener('scroll', () => {
      this.hideTooltip();
    });
  }

  showTooltip(element, event) {
    clearTimeout(this.hideTimeout);
    
    const definition = element.getAttribute('data-definition');
    if (!definition) return;

    this.lastMouseEvent = event;

    this.hoverTimeout = setTimeout(() => {
      this.tooltip.textContent = definition;
      this.positionTooltip(element, event);
      this.tooltip.classList.add('visible');
    }, 300); // 300ms delay
  }

  positionTooltip(element, mouseEvent) {
    const rect = element.getBoundingClientRect();
    const tooltipRect = this.tooltip.getBoundingClientRect();
    let centerX = rect.left + (rect.width / 2);

    if (rect.width > 200 && mouseEvent) {
      centerX = mouseEvent.clientX;
    }

    let top = rect.top - tooltipRect.height - 10;
    let left = centerX - (tooltipRect.width / 2);
    let showBelow = false;

    if (left < 10) {
      left = 10;
    } else if (left + tooltipRect.width > window.innerWidth - 10) {
      left = window.innerWidth - tooltipRect.width - 10;
    }

    if (top < 10) {
      top = rect.bottom + 10;
      showBelow = true;
    }

    this.tooltip.setAttribute('data-position', showBelow ? 'below' : 'above');

    this.tooltip.style.top = `${top + window.scrollY}px`;
    this.tooltip.style.left = `${left}px`;
  }

  hideTooltip() {
    clearTimeout(this.hoverTimeout);
    this.hideTimeout = setTimeout(() => {
      this.tooltip.classList.remove('visible');
    }, 100);
  }

  escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  escapeHtml(string) {
    const div = document.createElement('div');
    div.textContent = string;
    return div.innerHTML;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const glossaryManager = new GlossaryManager();
  glossaryManager.init();
});
