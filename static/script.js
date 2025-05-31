function analyzeText() {
  var text = document.getElementById("text").value;
  if (text) {
      fetch('/analyze', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify({ text: text })
      })
      .then(response => response.json())
      .then(data => {
          var resultDiv = document.getElementById("result");
          resultDiv.textContent = 'Sentiment: ' + data.sentiment;
      })
      .catch(error => console.error('Error:', error));
  } else {
      alert("Please enter some text to analyze.");
  }
}
