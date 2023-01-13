import 'package:flutter/material.dart';

/// TODO
class ShopController extends ChangeNotifier {
  // MEMBER VARIABLES //
  // loading dialog
  late Map<String, int> _cart; // TODO

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  /// Constructs a new [VehicleController].
  ShopController() : _cart = {};

  // //////////////// //
  // GETTING THE CART //
  // //////////////// //

  /// Returns the current cart within the controller
  Map<String, int> get cart {
    return _cart;
  }

  // ///////////////// //
  // UPDATING THE CART //
  // ///////////////// //

  /// Sets the cart into the controller.
  set cart(Map<String, int> cart) {
    // updating the cart
    _cart = cart;

    // notifying listeners
    notifyListeners();
  }

  /// Increments the quantity of the sign in the cart by one.
  void incrementSignQuantity(String sign) {
    // updating the cart
    if (_cart[sign] != null) {
      _cart[sign] = _cart[sign]! + 1;
    }

    // notifying listeners
    notifyListeners();
  }

  /// Decrements the quantity of the sign in the cart by one.
  void decrementSignQuantity(String sign) {
    // updating the cart
    if (_cart[sign] != null) {
      if (_cart[sign]! == 1) {
        _cart.remove(sign);
      } else {
        _cart[sign] = _cart[sign]! - 1;
      }
    }

    // notifying listeners
    notifyListeners();
  }

  /// Removes the given sign from the cart
  void removeSignFromCart(String sign) {
    // removing the sign from the cart
    _cart.remove(sign);

    // notifying listeners
    notifyListeners();
  }

  // /////////////////// //
  // SUBMITTING AN ORDER //
  // /////////////////// //

  /// Submits an order for the items currently in the cart.
  ///
  /// Just a dummy implementation that waits 2 seconds and returns.
  Future<void> submitOrder() async {
    // submiting order
    await Future.delayed(const Duration(seconds: 2));

    // clearing the cart
    _cart.clear();
  }
}
