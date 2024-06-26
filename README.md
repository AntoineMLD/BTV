# Découverte de la Computer Vision avec YOLO

## Contexte

---

Vous êtes IA engineer chez **InnoTech Solutions**, une entreprise innovante dans le domaine de la technologie, mais qui ne se spécialise pas encore dans la vision par ordinateur. Votre mission est d'aider à acculturer les collaborateurs de l'entreprise au transfert de compétences (transfer learning) et à l'utilisation de modèles de détection d'objets.

Pour ce faire, vous développerez une preuve de concept (POC) pour une application capable de détecter et d'identifier en temps réel des objets spécifiques via une caméra. Vous utiliserez des modèles de la famille YOLO (You Only Look Once), réputés pour leur efficacité en détection d'objets.

Afin de réaliser ce POC, vous devrez d'abord explorer les datasets disponibles sur [Roboflow](https://roboflow.com/) pour identifier une problématique pertinente que votre application pourrait résoudre. Cela vous permettra de choisir un sujet qui correspond aux intérêts et aux besoins diversifiés d'InnoTech Solutions, qu'il s'agisse de détection d'objets dans un contexte industriel, de suivi d'éléments dans des vidéos, ou de toute autre application pertinente.

L’objectif est de sélectionner un jeu de données afin de fine-tune un modèle de la famille YOLO (ne pas sélectionner un modèle directement).

## Objectifs Pédagogiques

---

À la fin de ce brief, vous serez capables de :

1. Identifier et définir une problématique de vision par ordinateur en explorant les datasets disponibles sur [Roboflow](https://roboflow.com/), adaptée aux besoins d'InnoTech Solutions.
2. Choisir une version de YOLO (par exemple, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10) en fonction de la disponibilité et de la pertinence pour votre problématique. Vous pouvez consulter la [documentation de YOLO](https://docs.ultralytics.com/) pour plus de détails.
3. Entraîner le modèle YOLO sélectionné sur un dataset pertinent de Roboflow, en tenant compte des spécificités de la détection d'objets pour votre application choisie.
4. Intégrer le modèle YOLO dans une application web interactive utilisant [Gradio](https://gradio.app/) ou [Streamlit](https://streamlit.io/), capable de se connecter à la webcam de votre ordinateur pour des prédictions en temps réel.
5. Réaliser une démonstration de votre application, avec une attention particulière à son fonctionnement en conditions réelles, y compris une démonstration depuis un smartphone pour prouver sa portabilité et son efficacité.

## Livrable

---

Pour ce brief, vous devrez fournir :

1. **Code de l'application** : Le code complet de l'application web intégrant le modèle YOLO. Assurez-vous que le code soit bien documenté et facile à comprendre.
2. **Démonstration fonctionnelle** : Une démonstration de l'application montrant la capacité du modèle à effectuer des prédictions en temps réel via la webcam. La démonstration devra illustrer comment l'application peut être utilisée pour la détection d'objets dans le contexte choisi.
3. **(Optionnel) Démonstration smartphone** : Une démonstration supplémentaire depuis un smartphone montrant la portabilité et l'efficacité de la solution développée, renforçant ainsi son utilité dans divers scénarios.

## Modalités pédagogiques

---

- **Groupe de** : 4 personnes
- **Durée** : Environ 2 semaines (rendu : 5 juillet)

---

## Instructions

1. **Exploration des datasets** : Visitez [Roboflow](https://roboflow.com/) et explorez les différents jeux de données disponibles. Choisissez un dataset qui répond à une problématique pertinente pour InnoTech Solutions.
2. **Choix du modèle YOLO** : Basé sur la documentation de [YOLO](https://docs.ultralytics.com/), choisissez la version du modèle qui convient le mieux à votre problématique.
3. **Entraînement du modèle** : Utilisez le dataset sélectionné pour entraîner le modèle YOLO choisi. Documentez le processus et les ajustements nécessaires pour optimiser les performances du modèle.
4. **Développement de l'application** : Créez une application web interactive utilisant [Gradio](https://gradio.app/) ou [Streamlit](https://streamlit.io/). Intégrez le modèle YOLO entraîné et configurez l'application pour se connecter à la webcam et effectuer des prédictions en temps réel.
5. **Démonstration** : Préparez une démonstration de votre application, montrant ses capacités en conditions réelles. Si possible, réalisez une démonstration supplémentaire utilisant un smartphone.

---

### Ressources Utiles

- [Roboflow](https://roboflow.com/)
- [Documentation YOLO](https://docs.ultralytics.com/)
- [Gradio](https://gradio.app/)
- [Streamlit](https://streamlit.io/)

---

Nous vous souhaitons bonne chance dans la réalisation de cette preuve de concept et espérons que cette expérience enrichira vos compétences en vision par ordinateur et en développement de modèles de détection d'objets.