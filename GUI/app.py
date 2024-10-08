from flask import Flask, render_template, request, redirect, url_for
import model
from model import all_predictions

app = Flask(__name__)
IMAGES_PER_PAGE = 20


# @app.route('/')
# @app.route('/<int:page>')
# def index(page=1):
#     predictions = model.get_paginated_predictions(page, IMAGES_PER_PAGE)
#     total_images = model.count_images()
#     total_pages = (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
#     return render_template('index.html', predictions=predictions, current_page=page, total_pages=total_pages)


@app.route('/')
@app.route('/<int:page>')
def index(page=1):
    # Use get_paginated_predictions to filter and paginate wrong predictions (those in 'unknown')
    predictions, total_images = model.get_paginated_predictions(
        page, IMAGES_PER_PAGE, class_filter='unknown')

    total_pages = (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    return render_template('index.html', predictions=predictions, current_page=page, total_pages=total_pages)

# @app.route('/')
# @app.route('/<int:page>')
# def index(page=1):
#     wrong_predictions = [p for p in all_predictions if p['wrong']]
#     print(wrong_predictions)
#     total_images = len(wrong_predictions)
#     total_pages = (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
#     predictions = wrong_predictions[(
#         page - 1) * IMAGES_PER_PAGE:page * IMAGES_PER_PAGE]

#     return render_template('index.html', predictions=predictions, current_page=page, total_pages=total_pages)


@app.route('/class/<class_name>/')
@app.route('/class/<class_name>/<int:page>')
def show_class(class_name, page=1):
    predictions, total_images = model.get_paginated_predictions(
        page, IMAGES_PER_PAGE, class_filter=class_name)
    total_pages = (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    return render_template('index.html', predictions=predictions, current_page=page, total_pages=total_pages, class_name=class_name)


# @app.route('/update', methods=['POST'])
# def update():
#     image_id = request.form['image_id']
#     prv_class = request.form['prv_class']
#     new_class = request.form['class']
#     model.update_label(image_id, prv_class, new_class)
#     return redirect(request.referrer or url_for('index'))

@app.route('/update', methods=['POST'])
def update():
    image_id = request.form['image_id']
    prv_class = request.form['prv_class']
    new_class = request.form['class']
    model.update_label(image_id, prv_class, new_class)
    return redirect(url_for('index'))


if __name__ == '__main__':
    model.initialize_predictions()  # Initialize predictions at startup
    app.run(debug=False)
