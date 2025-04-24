document.addEventListener('DOMContentLoaded', function() {
  const images = document.querySelectorAll('.post-content img:not(.no-expand)');
  
  const overlay = document.createElement('div');
  overlay.className = 'image-overlay';
  document.body.appendChild(overlay);
  
  const expandedContainer = document.createElement('div');
  expandedContainer.className = 'expanded-image-container';
  document.body.appendChild(expandedContainer);
  
  let isTransitioning = false;
  
  images.forEach(image => {
    // Add indicator that image is expandable
    image.classList.add('expandable');
    
    image.style.cursor = 'pointer';
    
    image.addEventListener('click', function(e) {
      if (isTransitioning) {
        e.preventDefault();
        return false;
      }
      
      expandedContainer.innerHTML = `
        <div class="solid-background">
          <a href="${this.src}" target="_blank" class="image-link" title="Open in new tab">
            <img src="${this.src}" alt="${this.alt || 'Expanded image'}" class="expanded-img">
          </a>
        </div>
      `;
      
      isTransitioning = true;
      overlay.classList.add('active');
      expandedContainer.classList.add('active');
      
      setTimeout(() => {
        isTransitioning = false;
      }, 300);
      
      document.body.style.overflow = 'hidden';
    });
  });
  
  overlay.addEventListener('click', function(e) {
    if (e.target === overlay) {
      closeExpanded();
    }
  });
  
  function closeExpanded() {
    isTransitioning = true;
    overlay.classList.remove('active');
    expandedContainer.classList.remove('active');
    document.body.style.overflow = '';
    
    overlay.addEventListener('transitionend', function cleanUp() {
      overlay.removeEventListener('transitionend', cleanUp);
      isTransitioning = false;
    });
  }
});
