import gradio as gr
from azure.core.credentials import AzureKeyCredential
from azure.maps.search import MapsSearchClient
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Créer le client Azure Maps
credential = AzureKeyCredential(os.getenv('AZURE_MAPS_ACCOUNT_KEY'))
search_client = MapsSearchClient(credential=credential)

# Définir la fonction de recherche
def search_address(latitude, longitude):
    # Construire la requête de recherche
    search_request = {
        "query": f"{latitude}, {longitude}",
        "maxResults": 1
    }

    # Envoyer la requête de recherche
    search_response = search_client.search(search_request)

    # Extraire l'adresse du premier résultat
    if search_response.results:
        result = search_response.results[0]
        address = result['address']['formattedAddress']
        return address
    else:
        return "Aucune adresse trouvée"

# HTML content to embed Azure Maps
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Azure Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />
    <script src="https://atlas.microsoft.com/sdk/js/atlas.min.js?api-version=2.0"></script>
    <link rel="stylesheet" href="https://atlas.microsoft.com/sdk/css/atlas.min.css?api-version=2.0" type="text/css" />
    <style>
        html, body, #map {
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
        }
    </style>
    <script type="text/javascript">
        var map;
        function GetMap() {
            map = new atlas.Map('map', {
                center: [0, 0], // Initial center position (longitude, latitude)
                zoom: 2,
                view: 'Auto',
                authOptions: {
                    authType: 'subscriptionKey',
                    subscriptionKey: os.getenv('AZURE_MAPS_ACCOUNT_KEY')
                }
            });

            map.events.add('click', function (e) {
                var position = e.position;
                var latitude = position[1];
                var longitude = position[0];

                // Update the hidden input fields with the clicked coordinates
                document.getElementById('latitude').value = latitude;
                document.getElementById('longitude').value = longitude;

                // Submit the form to call the Gradio function
                document.getElementById('coords-form').submit();
            });
        }
    </script>
</head>
<body onload="GetMap()">
    <div id="map"></div>
    <form id="coords-form" method="post" action="/predict">
        <input type="hidden" id="latitude" name="latitude">
        <input type="hidden" id="longitude" name="longitude">
    </form>
</body>
</html>
"""

# Créer l'interface Gradio
def display_address(latitude, longitude):
    return search_address(latitude, longitude)

with gr.Blocks() as demo:
    gr.HTML(html_content.replace('YOUR_AZURE_MAPS_KEY', os.getenv('AZURE_MAPS_ACCOUNT_KEY')))
    address_block = gr.Textbox(label="Adresse")
    latitude_input = gr.Textbox(visible=False)
    longitude_input = gr.Textbox(visible=False)
    btn = gr.Button("Obtenir l'adresse")

    # Définir la fonction pour obtenir l'adresse sur le clic
    def get_address(lat, lon):
        return search_address(lat, lon)

    btn.click(get_address, inputs=[latitude_input, longitude_input], outputs=address_block)

demo.launch()
