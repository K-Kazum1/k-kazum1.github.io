document.addEventListener('DOMContentLoaded', function() {
  // Process all yonaka quotes
  const quotes = document.querySelectorAll('.yonaka-quote, .yonaka-quote-custom');
  
  quotes.forEach(quote => {
    // Check if it's a custom quote with image data
    const imagePath = quote.getAttribute('data-image');
    const artistName = quote.getAttribute('data-artist');
    const artistLink = quote.getAttribute('data-artist-link');
    
    // Make sure all quotes have the base class
    quote.classList.add('yonaka-quote');
    
    // Wrap the quote text in a div for better centering
    const textContent = quote.innerHTML;
    const textDiv = document.createElement('div');
    textDiv.className = 'yonaka-quote-text';
    textDiv.innerHTML = textContent;
    
    // Clear and rebuild the quote structure
    quote.innerHTML = '';
    quote.appendChild(textDiv);
    
    // If there's a custom image, apply it
    if (imagePath) {
      quote.classList.add('custom-image');
      
      // Create a unique ID and style for this quote
      const quoteId = 'quote-' + Math.random().toString(36).substr(2, 9);
      quote.id = quoteId;
      
      const style = document.createElement('style');
      style.textContent = `
        #${quoteId}:before {
          background-image: url('${imagePath}');
        }
      `;
      document.head.appendChild(style);
    }
    
    // Add artist attribution if provided
    if (artistName) {
      const attribution = document.createElement('span');
      attribution.className = 'artist-credit';
      
      if (artistLink) {
        attribution.innerHTML = `Art by <a href="${artistLink}" target="_blank">${artistName}</a>`;
      } else {
        attribution.textContent = `Art by ${artistName}`;
      }
      
      quote.appendChild(attribution);
    }
  });
  
  // Process blockquotes with Yonaka syntax
  const blockquotes = document.querySelectorAll('blockquote');
  blockquotes.forEach(quote => {
    const firstP = quote.querySelector('p:first-child');
    if (firstP) {
      const text = firstP.textContent;
      
      // Regular Yonaka quote
      if (text.startsWith('[Yonaka]:')) {
        quote.classList.add('yonaka-quote');
        firstP.textContent = text.replace('[Yonaka]:', '').trim();
        
        // Wrap the content similar to other quotes
        const textContent = quote.innerHTML;
        const textDiv = document.createElement('div');
        textDiv.className = 'yonaka-quote-text';
        textDiv.innerHTML = textContent;
        
        quote.innerHTML = '';
        quote.appendChild(textDiv);
      }
      // Mint quotes (keeping the same pattern)
      else if (text.startsWith('[Mint]:')) {
        // Similar processing for Mint quotes
        // ...
      }
    }
  });
});
