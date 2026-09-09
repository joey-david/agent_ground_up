from template import render

def test_render():
    assert render('hi {{name}}', {'name': 'ada'}) == 'hi ada'
    assert render('{{a}}-{{b}}', {'a': '1', 'b': '2'}) == '1-2'
    assert render('hi {{who}}', {}) == 'hi {{who}}'
    assert render('no placeholders', {'x': 'y'}) == 'no placeholders'
