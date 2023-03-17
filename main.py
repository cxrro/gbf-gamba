import numpy as np
# import pytesseract
import pyautogui
from collections import Counter
import itertools
import cv2
import random
import os
import glob
import time

# Configure the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = '/System/Volumes/Data/opt/homebrew/Cellar/tesseract/5.3.0_1/bin/tesseract'


# Capture the screen
def capture_screen(region=None):
    screenshot = pyautogui.screenshot()
    if region:
        screenshot = screenshot.crop(region)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


# Preprocess the image for contour detection
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(
        gray_image, 128, 255, cv2.THRESH_BINARY)
    return gray_image, thresholded_image


# Find card contours
def find_card_contours(image):
    preprocessed_image = preprocess_image(image)
    contours, _ = cv2.findContours(
        preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Filter card contours based on size and aspect ratio
def filter_card_contours(contours, min_area=100, max_area=90000, min_aspect_ratio=0, max_aspect_ratio=2):
    card_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h)

        if min_area <= area <= max_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            card_contours.append((x, y, w, h))

    return card_contours


# Detect the positions of the playing cards
def detect_card_positions(screen):
    gray_image, preprocessed_image = preprocess_image(screen)
    # Save the thresholded image
    # cv2.imwrite('gray_image.png', gray_image)
    # cv2.imwrite('thresholded_image.png', preprocessed_image)
    contours, _ = cv2.findContours(
        preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the card contours
    card_positions = filter_card_contours(contours)
    return card_positions


# Load card value templates
def load_templates():
    templates = {}
    for file in glob.glob("templates/*.png"):
        name = os.path.splitext(os.path.basename(file))[0]
        templates[name] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return templates


# Match card value using template matching
def match_card_value(card_image, templates, threshold=0.9):
    card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)

    # Crop the card_gray image to the top-left corner
    cropped_card_gray = card_gray[0:50, 0:40]
    for value, template in templates.items():
        res = cv2.matchTemplate(
            cropped_card_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > threshold:
            return value
    return None


def match_card_value_double(card_image, templates, threshold=0.9):
    for value, template in templates.items():
        res = cv2.matchTemplate(
            card_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > threshold:
            return value
    return None


# Extract card values using template matching
def extract_card_values(card_images, templates, threshold):
    card_values = []
    for img in card_images:
        matched_value = match_card_value(img, templates, threshold)
        if matched_value is not None:
            card_values.append(matched_value)
        else:
            print("No match found for card value")
    return card_values


def evaluate_poker_hand(hand):
    values = [v for v in hand if v]  # Ignore empty strings

    value_counts = Counter(values)

    value_ranks = {str(i): i for i in range(2, 11)}
    value_ranks.update({"10": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "Z": 15})

    sorted_values = sorted(values, key=lambda x: value_ranks[x])
    straight = all([value_ranks[sorted_values[i]] ==
                   value_ranks[sorted_values[i-1]] + 1 for i in range(1, len(sorted_values))])

    if straight:
        # fix this for ace low straight
        # also i think it doesnt work ever
        return 4
    elif any(count == 4 for count in value_counts.values()):
        return 20
    elif any(count == 3 for count in value_counts.values()):
        if any(count == 2 for count in value_counts.values()):  # Full house
            return 10
        else:  # 3 of a kind
            return 1
    elif sum(1 for count in value_counts.values() if count == 2) == 2:  # 2 pairs
        return 1
    return 0


def calculate_expected_value(kept_cards, discarded_cards):
    base_deck = ['2', '3', '4', '5', '6', '7',
                 '8', '9', '10', 'J', 'Q', 'K', 'A'] * 4 + ['Z']
    # 'Z' is a joker

    # Make a copy of the base_deck before removing cards
    deck = base_deck.copy()
    for card in kept_cards:
        card_str = card
        if card_str not in deck:
            print(f"Card not found in deck: {card_str}")
            print(f"Kept cards: {kept_cards}")
            continue
        deck.remove(card_str)

    for card in kept_cards:
        card_str = card
        if card_str not in deck:
            print(f"Card not found in deck: {card_str}")
            print(f"Kept cards: {kept_cards}")
            continue
        deck.remove(card_str)

    num_simulations = 1000
    total_score = 0

    for _ in range(num_simulations):
        # Draw new cards for the discarded cards
        drawn_cards = random.sample(deck, len(discarded_cards))

        # Create the new hand
        new_hand = kept_cards + drawn_cards

        # Evaluate the new hand
        score = evaluate_poker_hand(new_hand)
        # print(f'hand: {new_hand} score: {score}')

        total_score += score

    # Calculate the average score (expected value) of the hand after mulligan
    expected_value = total_score / (num_simulations/10)
    return expected_value


def decide_mulligan(hand):
    best_score = -1
    best_combination = []

    # Create a copy of the hand without the joker
    hand_without_joker = [card for card in hand if card != 'Z']

    # Iterate through all possible combinations of cards to keep (0 to 4 cards if joker is present, otherwise 0 to 5 cards)
    max_cards_to_keep = 4 if 'Z' in hand else 5
    for num_cards_to_keep in range(max_cards_to_keep + 1):
        for combination in itertools.combinations(hand_without_joker, num_cards_to_keep):
            # Reconstruct the kept_cards list with the joker in its original position
            kept_cards = [card if card in combination or card ==
                          'Z' else None for card in hand]
            kept_cards = [card for card in kept_cards if card is not None]

            discarded_cards = [card for card in hand if card not in kept_cards]

            # Calculate the expected value of the current combination
            expected_value = calculate_expected_value(
                kept_cards, discarded_cards)

            # Print the current combination and its expected value
            print(
                f"Kept cards: {kept_cards}, Discarded cards: {discarded_cards}, Expected value: {expected_value}")

            # Update the best combination if the current expected value is higher
            if expected_value > best_score:
                best_score = expected_value
                best_combination = kept_cards

    # Get the indices of the cards to mulligan
    mulligan_indices = [i for i, card in enumerate(
        hand) if card not in best_combination]

    return mulligan_indices


def click(button):
    y = 1460/2
    if button == "ok":
        x = 760/2
    elif button == "left":
        x = 600/2
    elif button == "right":
        x = 410
    pyautogui.click(x, y)


def detect_single_card(region, templates, threshold=0.5):

    # Extract card image
    card_image = capture_screen(region)

    # Crop the top left corner of the card image
    cropped_corner = card_image[0:70, 0:56]

    # resize to match template size
    scale = 5/7
    resized_corner = cv2.resize(
        cropped_corner, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray_corner = card_gray = cv2.cvtColor(
        cropped_corner, cv2.COLOR_BGR2GRAY)
    processed_corner = high_contrast_image = cv2.addWeighted(
        gray_corner, 1.6, gray_corner, 0, 0)
    cv2.imwrite('processed_corner.png', processed_corner)

    # Get the value of the card
    value = match_card_value_double(processed_corner, templates, threshold)

    return value


def value_to_int(card_value):
    face_cards = {'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    if card_value.isdigit():
        return int(card_value)
    elif card_value in face_cards:
        return face_cards[card_value]
    else:
        raise ValueError(f"Invalid card value: {card_value}")


def is_double_up_mode(double_up_template, threshold=0.8):
    mode_region = (140*2, 150*2, 640*2, 250*2)
    screen = capture_screen(mode_region)
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(
        gray_screen, double_up_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)

    return max_val > threshold


def play_double_up(double_up_template, templates):
    card1_region = (242*2, 402*2, 360*2, 570*2)
    # card2_region = (401*2, 401*2, 520*2, 570*2)

    card1_value = detect_single_card(card1_region, templates)
    # card2_value = detect_single_card(card2_region, templates)

    if card1_value:
        value = value_to_int(card1_value)
        # Decide whether to click higher or lower
        if value > 7:
            print(str(value) + " lower")
            click("right")
        else:
            print(str(value) + " higher")
            click("left")
        # wait for result
        time.sleep(1)
        click("ok")
        if not is_double_up_mode(double_up_template):
            print("i lost :-(")
            main()
    else:
        print("couldn't read card 1 value")

    def decide_play_again():
        pass


# Main function
def main():
    left = 240
    top = 850
    table_region = (left, top, 1300, 1200)
    money_region = (220, 1240, 1300, 1400)

    # Capture the screen
    screen = capture_screen(table_region)

    # Detect card positions
    card_positions = detect_card_positions(screen)
    print(f'Detected card positions: {card_positions}')

    # Sort card positions from left to right
    card_positions.sort(key=lambda pos: pos[0])

    # Extract card images and suits
    card_images = [screen[y:y+h, x:x+w] for x, y, w, h in card_positions]

    # Load the templates and extract card values
    templates = load_templates()
    # Extract card values with a loop for lowering the threshold
    threshold = 1.0
    card_values = []
    while len(card_values) < 5 and threshold >= 0:
        card_values = extract_card_values(card_images, templates, threshold)
        threshold -= 0.05

    print(f'Card values: {card_values}')

    # Draw bounding boxes around the detected cards
    for i, ((x, y, w, h), value) in enumerate(zip(card_positions, card_values)):
        cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(screen, f"{value}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Save individual card images
        cv2.imwrite(f'card_{i}.png', card_images[i])

    # Save the image with bounding boxes
    cv2.imwrite('detected_cards.png', screen)

    if len(card_values) == 5:
        # Decide which cards to mulligan
        mulligan_indices = decide_mulligan(card_values)
        print(f'Mulligan indices: {mulligan_indices}')

        # Focus the window by clicking the top-left corner of the first card
        first_card_x, first_card_y, _, _ = card_positions[0]
        pyautogui.click((left/2) + first_card_x/2, (top/2) + first_card_y/2)

        # Click on the cards to keep
        for index, (x, y, w, h) in enumerate(card_positions):
            if index not in mulligan_indices:
                # Calculate the click coordinates
                click_x = (left + (x + w // 2))/2
                click_y = (top + (y + h // 2))/2

                print(f"Click attempt at coordinates: ({click_x}, {click_y})")
                pyautogui.click(click_x, click_y)
        # wait for the cards to be selected
        time.sleep(1)
        # Click OK
        click("ok")
        # click deal or yes
        time.sleep(3)
        click("right")
    else:
        print("Not enough cards detected. Only found " + str(len(card_values)))

    # Load the double up mode template
    double_up_template = cv2.imread(
        'double_up_template.png', cv2.IMREAD_GRAYSCALE)

    time.sleep(2)
    # Check for double up mode
    if is_double_up_mode(double_up_template):
        print("Playing double up...")
        play_double_up(double_up_template, templates)
    else:
        print("Not in double up mode")
        main()


if __name__ == '__main__':
    main()
