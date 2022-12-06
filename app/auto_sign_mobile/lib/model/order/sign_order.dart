/// An order for a specific sign within the system.
///
/// Contains the sign being ordered and its quantity.
class SignOrder {
  // MEMBERS //
  String signTitle; // the title of the sign being ordered
  int quantity; // the quantity of the sign being ordered

  // ///////////////// //
  // CLASS CONSTRUCTOR //
  // ///////////////// //

  SignOrder({
    this.signTitle = "",
    this.quantity = 0,
  });

  // ///////////////// //
  // CHANGING QUANTITY //
  // ///////////////// //

  /// Increments the quantity of the sign being ordered by 1.
  void increment() {
    quantity++;
  }

  /// Decrements the quantity of the sign being ordered by 1.
  void decrement() {
    quantity--;
  }
}
