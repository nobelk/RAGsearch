import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:app_ui/main.dart';
import 'package:app_ui/services/search_stream_service.dart';

void main() {
  testWidgets('App renders chat screen', (WidgetTester tester) async {
    final service = SearchStreamService(baseUrl: 'http://localhost:8000');
    await tester.pumpWidget(DocSearchApp(service: service));

    expect(find.text('Document Search'), findsOneWidget);
    expect(find.byType(TextField), findsOneWidget);
  });
}
